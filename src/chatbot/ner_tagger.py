"""PhoBERT NER Tagger — Head 2.

Fine-tunes vinai/phobert-base for Vietnamese legal Named Entity Recognition.
9 entity types with BIO tagging scheme (19 tags total).
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from src.common.config import NER_LABELS, ENTITY_TYPES, PHOBERT_MODEL, MAX_SEQ_LENGTH


class PhoBERTNERTagger(nn.Module):
    """Named Entity Recognition head on top of PhoBERT.

    Architecture: PhoBERT → Dropout → Linear(768, num_labels)

    Entity types: LUAT, THONG_TU, NGHI_DINH, DIEU, KHOAN, DIEM, CO_QUAN, KHAISUAT, NGAY_THANG
    Tagging scheme: BIO (B-entity, I-entity, O)

    Input: Vietnamese legal text (word-segmented)
    Output: Token-level entity tags
    """

    def __init__(self, model_name: str = PHOBERT_MODEL, num_labels: int = None):
        super().__init__()
        self.num_labels = num_labels or len(NER_LABELS)
        self.phobert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, self.num_labels)

        self.id2label = {i: label for i, label in enumerate(NER_LABELS[:self.num_labels])}
        self.label2id = {label: i for i, label in enumerate(NER_LABELS[:self.num_labels])}

    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass.

        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            labels: BIO tag IDs (batch, seq_len), -100 for subword tokens

        Returns:
            Loss (if labels provided) and logits
        """
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (batch, seq_len, 768)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)  # (batch, seq_len, num_labels)

        loss = None
        if labels is not None:
            # Ignore -100 labels in loss computation
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # Flatten for cross entropy
            active_logits = logits.view(-1, self.num_labels)
            active_labels = labels.view(-1)
            loss = loss_fct(active_logits, active_labels)

        return {"loss": loss, "logits": logits}

    def predict(self, text: str, tokenizer, device: str = "cpu") -> list[dict]:
        """Predict entities in a single text.

        Args:
            text: Vietnamese legal text (should be pre-segmented)
            tokenizer: PhoBERT tokenizer
            device: 'cpu' or 'cuda'

        Returns:
            List of dicts: [{"entity": "THONG_TU", "text": "Thông tư 99", "start": 0, "end": 10}, ...]
        """
        self.eval()
        self.to(device)

        encoding = tokenizer(
            text, max_length=MAX_SEQ_LENGTH, truncation=True,
            padding="max_length", return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            output = self.forward(input_ids, attention_mask)
            predictions = torch.argmax(output["logits"], dim=-1)[0]

        # Convert token-level predictions to entities
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
                entity_type = label[2:]
                current_entity = {
                    "entity": entity_type,
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