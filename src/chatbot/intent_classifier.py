import torch
import torch.nn as nn
from transformers import AutoModel
from src.common.config import INTENT_LABELS, PHOBERT_MODEL, MAX_SEQ_LENGTH


class PhoBERTIntentClassifier(nn.Module):
    def __init__(self, model_name: str = PHOBERT_MODEL, num_intents: int = None):
        super().__init__()
        self.num_intents = num_intents or len(INTENT_LABELS)
        self.phobert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, self.num_intents)
        self.id2label = {i: label for i, label in enumerate(INTENT_LABELS[:self.num_intents])}
        self.label2id = {label: i for i, label in enumerate(INTENT_LABELS[:self.num_intents])}

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls_output))

    def predict(self, text: str, tokenizer, device: str = "cpu") -> dict:
        self.eval()
        self.to(device)
        encoding = tokenizer(
            text, max_length=MAX_SEQ_LENGTH, truncation=True, padding="max_length", return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, pred_idx].item()
        return {
            "intent": self.id2label[pred_idx],
            "confidence": confidence,
            "probabilities": {self.id2label[i]: probs[0, i].item() for i in range(self.num_intents)},
        }
