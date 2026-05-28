import torch
import torch.nn as nn
from transformers import AutoModel
from backend.common.config import NER_LABELS, PHOBERT_MODEL, MAX_SEQ_LENGTH


class PhoBERTNERTagger(nn.Module):
    def __init__(self, model_name: str = PHOBERT_MODEL, num_labels: int = None):
        super().__init__()
        self.num_labels = num_labels or len(NER_LABELS)
        self.phobert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, self.num_labels)
        self.id2label = {i: label for i, label in enumerate(NER_LABELS[:self.num_labels])}
        self.label2id = {label: i for i, label in enumerate(NER_LABELS[:self.num_labels])}

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return {"loss": loss, "logits": logits}

    def predict(self, text: str, tokenizer, device: str = "cpu") -> list:
        self.eval()
        self.to(device)
        encoding = tokenizer(
            text, max_length=MAX_SEQ_LENGTH, truncation=True, padding="max_length", return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        with torch.no_grad():
            output = self.forward(input_ids, attention_mask)
            predictions = torch.argmax(output["logits"], dim=-1)[0]
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        entities = []
        current_entity = None
        for idx, (token, pred_id) in enumerate(zip(tokens, predictions)):
            if token in tokenizer.special_tokens_map.values():
                continue
            if pred_id.item() == self.label2id.get("O", 0):
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
                continue
            label = self.id2label.get(pred_id.item(), "O")
            if label.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "entity": label[2:],
                    "text": token.replace("▁", "").replace("@@", ""),
                    "start": idx,
                    "end": idx + 1,
                }
            elif label.startswith("I-") and current_entity:
                current_entity["text"] += token.replace("▁", " ").replace("@@", "")
                current_entity["end"] = idx + 1
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        if current_entity:
            entities.append(current_entity)
        return entities
