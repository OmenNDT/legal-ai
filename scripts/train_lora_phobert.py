"""Train LoRA PhoBERT on Luật Kế toán 2025 for intent classification + NER.

Uses PEFT LoRA adapters on PhoBERT base, fine-tuning both intent and NER heads.
Dataset: data/processed/qa_ke_toan_train.json (generated from law text)

Usage:
    PYTHONPATH=/path/to/legal-ai:$PYTHONPATH python scripts/train_lora_phobert.py
"""

import json
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from pathlib import Path
from tqdm import tqdm

from src.common.config import (
    PHOBERT_MODEL, MAX_SEQ_LENGTH,
    INTENT_LABELS, NER_LABELS,
    LEARNING_RATE, NUM_EPOCHS, TRAIN_BATCH_SIZE,
    INTENT_MODEL_DIR, NER_MODEL_DIR,
)


class LoRAConfig:
    """LoRA hyperparameters."""
    R = 16
    ALPHA = 32
    DROPOUT = 0.1
    TARGET_MODULES = ["query", "key", "value", "dense"]


class LawQADataset(Dataset):
    """Dataset for legal QA with intent + NER labels."""

    def __init__(self, data_path: Path, tokenizer, intent_label2id, ner_label2id, max_length=256):
        with open(data_path, encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.intent_label2id = intent_label2id
        self.ner_label2id = ner_label2id
        self.max_length = max_length
        self.num_intents = len(intent_label2id)
        self.num_ner_labels = len(ner_label2id)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"]
        answer = item["answer"]
        intent = item["intent"]
        ner_labels = item["ner_labels"]

        # Combine question + answer for NER (answer contains entities)
        text = f"{question} [SEP] {answer}"

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Intent label
        intent_label = self.intent_label2id.get(intent, 0)

        # NER labels aligned with tokenizer tokens
        # For simplicity, map word-level NER to token-level by extending
        tokens = text.split()
        token_ner_labels = []
        for i, token in enumerate(tokens):
            wordpiece_tokens = self.tokenizer.tokenize(token)
            if i < len(ner_labels):
                label = ner_labels[i]
            else:
                label = "O"
            token_ner_labels.extend([label] + ["O"] * (len(wordpiece_tokens) - 1))

        # Pad/truncate to max_length
        token_ner_ids = [self.ner_label2id.get(l, self.ner_label2id["O"]) for l in token_ner_labels]
        if len(token_ner_ids) < self.max_length:
            token_ner_ids += [self.ner_label2id["O"]] * (self.max_length - len(token_ner_ids))
        else:
            token_ner_ids = token_ner_ids[:self.max_length]

        ner_labels_tensor = torch.tensor(token_ner_ids, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "intent_label": torch.tensor(intent_label, dtype=torch.long),
            "ner_labels": ner_labels_tensor,
        }


class PhoBERTMultiTaskLoRA(nn.Module):
    """PhoBERT with LoRA adapters + Intent + NER heads."""

    def __init__(self, num_intents: int, num_ner_labels: int, lora_config: LoRAConfig):
        super().__init__()
        self.phobert = AutoModel.from_pretrained(PHOBERT_MODEL)

        # Apply LoRA
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_config.R,
            lora_alpha=lora_config.ALPHA,
            lora_dropout=lora_config.DROPOUT,
            target_modules=lora_config.TARGET_MODULES,
            bias="none",
        )
        self.phobert = get_peft_model(self.phobert, peft_config)

        # Intent classification head
        self.intent_dropout = nn.Dropout(0.1)
        self.intent_classifier = nn.Linear(768, num_intents)

        # NER head
        self.ner_dropout = nn.Dropout(0.1)
        self.ner_classifier = nn.Linear(768, num_ner_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (batch, seq_len, 768)

        # Intent: use [CLS] token (index 0)
        cls_output = sequence_output[:, 0, :]  # (batch, 768)
        cls_output = self.intent_dropout(cls_output)
        intent_logits = self.intent_classifier(cls_output)  # (batch, num_intents)

        # NER: use all tokens
        ner_output = self.ner_dropout(sequence_output)
        ner_logits = self.ner_classifier(ner_output)  # (batch, seq_len, num_ner_labels)

        return intent_logits, ner_logits


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    intent_correct = 0
    intent_total = 0
    ner_correct = 0
    ner_total = 0

    intent_criterion = nn.CrossEntropyLoss()
    ner_criterion = nn.CrossEntropyLoss(ignore_index=-100)

    progress = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        intent_labels = batch["intent_label"].to(device)
        ner_labels = batch["ner_labels"].to(device)

        optimizer.zero_grad()
        intent_logits, ner_logits = model(input_ids, attention_mask)

        # Intent loss
        intent_loss = intent_criterion(intent_logits, intent_labels)

        # NER loss
        ner_logits_flat = ner_logits.view(-1, ner_logits.size(-1))
        ner_labels_flat = ner_labels.view(-1)
        ner_loss = ner_criterion(ner_logits_flat, ner_labels_flat)

        loss = intent_loss + ner_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        # Metrics
        intent_preds = torch.argmax(intent_logits, dim=-1)
        intent_correct += (intent_preds == intent_labels).sum().item()
        intent_total += intent_labels.size(0)

        ner_preds = torch.argmax(ner_logits, dim=-1)
        ner_mask = ner_labels != -100
        ner_correct += ((ner_preds == ner_labels) & ner_mask).sum().item()
        ner_total += ner_mask.sum().item()

        progress.set_postfix({
            "loss": f"{loss.item():.4f}",
            "intent_acc": f"{intent_correct/intent_total:.3f}",
            "ner_acc": f"{ner_correct/ner_total:.3f}" if ner_total > 0 else "0.000",
        })

    avg_loss = total_loss / len(dataloader)
    intent_acc = intent_correct / intent_total if intent_total > 0 else 0
    ner_acc = ner_correct / ner_total if ner_total > 0 else 0
    return avg_loss, intent_acc, ner_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Labels
    intent_label2id = {label: i for i, label in enumerate(INTENT_LABELS)}
    ner_label2id = {label: i for i, label in enumerate(NER_LABELS)}

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL)

    # Dataset
    data_path = Path("data/processed/qa_ke_toan_train_v2.json")
    dataset = LawQADataset(data_path, tokenizer, intent_label2id, ner_label2id, MAX_SEQ_LENGTH)
    dataloader = DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

    print(f"Dataset size: {len(dataset)}")
    print(f"Intent classes: {len(INTENT_LABELS)}")
    print(f"NER labels: {len(NER_LABELS)}")

    # Model
    lora_config = LoRAConfig()
    model = PhoBERTMultiTaskLoRA(
        num_intents=len(INTENT_LABELS),
        num_ner_labels=len(NER_LABELS),
        lora_config=lora_config,
    ).to(device)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(dataloader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    # Training loop
    best_loss = float("inf")
    for epoch in range(1, NUM_EPOCHS + 1):
        avg_loss, intent_acc, ner_acc = train_epoch(model, dataloader, optimizer, scheduler, device, epoch)
        print(f"Epoch {epoch}: loss={avg_loss:.4f}, intent_acc={intent_acc:.3f}, ner_acc={ner_acc:.3f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save checkpoint
            save_dir = Path("data/models/lora_ke_toan")
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "intent_label2id": intent_label2id,
                "ner_label2id": ner_label2id,
            }, save_dir / "best_model.pt")
            print(f"  Saved best model → {save_dir / 'best_model.pt'}")

    print("Training complete!")
    print(f"Best model saved at: data/models/lora_ke_toan/best_model.pt")


if __name__ == "__main__":
    main()
