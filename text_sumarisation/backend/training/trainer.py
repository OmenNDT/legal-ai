import inspect
from pathlib import Path
from typing import Any, List, cast
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from backend.utils.logger import Logger

# Wrapper fine-tune BART trên dữ liệu CUAD
class BartFineTuner:
    def __init__(
        self,
        model_name: str = "facebook/bart-large-cnn",
        output_dir: Path = Path("outputs/bart-cuad"),
        max_input: int = 1024,
        max_target: int = 256,
        epochs: int = 3,
        batch_size: int = 2,
        grad_accum: int = 8,
        lr: float = 3e-5,
        fp16: bool = True,
        use_lora: bool = False,
        early_stopping_patience: int = 2,
        early_stopping_threshold: float = 0.0
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.max_input = max_input
        self.max_target = max_target
        self.epochs = epochs
        self.batch_size = batch_size
        self.grad_accum = grad_accum
        self.lr = lr
        self.fp16 = fp16
        self.use_lora = use_lora
        # Số epoch eval_loss được phép không cải thiện trước khi dừng
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.log = Logger.get("trainer")

    # Tokenize input/target (API mới của transformers >= 4.22: dùng text_target)
    def _tokenize(self, examples, tokenizer):
        model_in = tokenizer(
            examples["input"],
            text_target = examples["target"],
            max_length = self.max_input,
            truncation = True
        )
        # Truncate label thủ công vì text_target không nhận max_target riêng
        model_in["labels"] = [ids[: self.max_target] for ids in model_in["labels"]]
        return model_in

    # Tiến hành fine-tune
    def fit(self, data: dict):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        # LoRA tuỳ chọn để giảm VRAM (rất khuyến nghị với BART-large trên 24GB)
        if self.use_lora:
            cfg = LoraConfig(
                task_type = TaskType.SEQ_2_SEQ_LM,
                r = 16,
                lora_alpha = 32,
                lora_dropout = 0.05,
                target_modules = ["q_proj", "v_proj"]
            )
            model = get_peft_model(model, cfg)
            model.print_trainable_parameters()

        ds_train = Dataset.from_list(data["train"])
        ds_val = Dataset.from_list(data["val"])
        ds_train = ds_train.map(lambda x: self._tokenize(x, tokenizer), batched = True, remove_columns = ds_train.column_names)
        ds_val = ds_val.map(lambda x: self._tokenize(x, tokenizer), batched = True, remove_columns = ds_val.column_names)

        collator = DataCollatorForSeq2Seq(tokenizer, model = model)
        args = Seq2SeqTrainingArguments(
            output_dir = str(self.output_dir),
            num_train_epochs = self.epochs,
            per_device_train_batch_size = self.batch_size,
            per_device_eval_batch_size = self.batch_size,
            gradient_accumulation_steps = self.grad_accum,
            learning_rate = self.lr,
            warmup_ratio = 0.05,
            fp16 = self.fp16,
            logging_steps = 20,
            save_strategy = "epoch",
            eval_strategy = "epoch",
            predict_with_generate = True,
            generation_max_length = self.max_target,
            save_total_limit = 2,
            # Tự load lại best checkpoint khi train xong, dùng eval_loss làm thước đo
            load_best_model_at_end = True,
            metric_for_best_model = "eval_loss",
            greater_is_better = False,
            report_to = "none"
        )
        # Early stopping callback: nếu eval_loss không giảm sau N lần eval thì dừng
        early_stop = EarlyStoppingCallback(
            early_stopping_patience = self.early_stopping_patience,
            early_stopping_threshold = self.early_stopping_threshold,
        )
        callbacks = cast(List[TrainerCallback], [early_stop])
        # transformers >= 4.46 đã đổi tên 'tokenizer' thành 'processing_class'
        tok_key = "processing_class" if "processing_class" in inspect.signature(Seq2SeqTrainer.__init__).parameters else "tokenizer"
        trainer = Seq2SeqTrainer(
            model = model,
            args = args,
            train_dataset = cast(Any, ds_train),
            eval_dataset = cast(Any, ds_val),
            data_collator = collator,
            callbacks = callbacks,
            **{tok_key: tokenizer}
        )
        trainer.train()
        trainer.save_model(str(self.output_dir / "final"))
        tokenizer.save_pretrained(str(self.output_dir / "final"))
        return str(self.output_dir / "final")
