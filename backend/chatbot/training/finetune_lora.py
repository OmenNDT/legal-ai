import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType

class JsonlSeq2SeqDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_input: int = 1024, max_target: int = 256):
        self._rows = [json.loads(l) for l in path.read_text(encoding = "utf-8").splitlines() if l.strip()]
        self._tok = tokenizer
        self._max_input = max_input
        self._max_target = max_target

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, idx):
        row = self._rows[idx]
        # Transformers >=4.22 dùng text_target thay cho as_target_tokenizer()
        model_inputs = self._tok(
            row["prompt"],
            max_length = self._max_input,
            truncation = True,
            padding = False
        )
        labels = self._tok(
            text_target = row["answer"] or "",
            max_length = self._max_target,
            truncation = True,
            padding = False
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

if __name__ == "__main__":
    p = argparse.ArgumentParser(description = "Fine-tune BARTpho-syllable với LoRA")
    p.add_argument("--train-file", type = Path, required = True)
    p.add_argument("--val-file", type = Path, required = True)
    p.add_argument("--output-dir", type = Path, required = True)
    p.add_argument("--base-model", default = "vinai/bartpho-syllable")
    p.add_argument("--epochs", type = int, default = 3)
    p.add_argument("--batch-size", type = int, default = 8)
    p.add_argument("--grad-accum", type = int, default = 2)
    p.add_argument("--lr", type = float, default = 3e-4)
    p.add_argument("--lora-r", type = int, default = 16)
    p.add_argument("--lora-alpha", type = int, default = 32)
    p.add_argument("--lora-dropout", type = float, default = 0.05)
    p.add_argument("--max-input", type = int, default = 1024)
    p.add_argument("--max-target", type = int, default = 256)
    p.add_argument("--warmup-ratio", type = float, default = 0.05)
    args = p.parse_args()

    print(f"[Finetune] base_model = {args.base_model}")
    print(f"[Finetune] train = {args.train_file} val = {args.val_file}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base = AutoModelForSeq2SeqLM.from_pretrained(
        args.base_model,
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    )

    # LoRA cho BART (encoder-decoder) — target các module attention
    lora_cfg = LoraConfig(
        task_type = TaskType.SEQ_2_SEQ_LM,
        r = args.lora_r,
        lora_alpha = args.lora_alpha,
        lora_dropout = args.lora_dropout,
        target_modules = ["q_proj", "v_proj", "k_proj", "out_proj"],
        bias = "none"
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()

    train_ds = JsonlSeq2SeqDataset(args.train_file, tokenizer, args.max_input, args.max_target)
    val_ds = JsonlSeq2SeqDataset(args.val_file, tokenizer, args.max_input, args.max_target)
    print(f"[Finetune] train_size = {len(train_ds)} val_size = {len(val_ds)}")

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    targs = Seq2SeqTrainingArguments(
        output_dir = str(args.output_dir),
        num_train_epochs = args.epochs,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        learning_rate = args.lr,
        warmup_ratio = args.warmup_ratio,
        logging_steps = 50,
        eval_strategy = "epoch",
        save_strategy = "epoch",
        save_total_limit = 2,
        load_best_model_at_end = True,
        metric_for_best_model = "eval_loss",
        greater_is_better = False,
        fp16 = torch.cuda.is_available(),
        report_to = "none",
        predict_with_generate = False
    )

    trainer = Seq2SeqTrainer(
        model = model,
        args = targs,
        train_dataset = train_ds,
        eval_dataset = val_ds,
        processing_class = tokenizer,
        data_collator = collator
    )

    trainer.train()
    final_dir = args.output_dir / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[Finetune] LoRA adapter đã lưu tại: {final_dir}")
    print(f"[Finetune] Set env LORA_WEIGHTS = {final_dir} để inference dùng adapter này")
