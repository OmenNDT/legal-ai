import torch
import torch.nn as nn
from transformers import AutoModel

class PhoBERTSentenceScorer(nn.Module):
    def __init__(self, model_name: str = "vinai/phobert-base", hidden_size: int = 768):
        super().__init__()
        self.phobert = AutoModel.from_pretrained(model_name)
        self.pre_classifier = nn.Linear(hidden_size * 3, hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.regression_head = nn.Linear(hidden_size, 1)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, sent_ids, sent_mask, doc_ids, doc_mask):
        sent_emb = self._mean_pooling(self.phobert(input_ids=sent_ids, attention_mask=sent_mask), sent_mask)
        doc_emb = self._mean_pooling(self.phobert(input_ids=doc_ids, attention_mask=doc_mask), doc_mask)
        combined = torch.cat([sent_emb, doc_emb, sent_emb * doc_emb], dim=1)
        x = torch.nn.ReLU()(self.pre_classifier(combined))
        return self.regression_head(self.dropout(x))

    def score_sentences(self, sentences: list, document: str, tokenizer, max_length: int = 256) -> list:
        self.eval()
        device = next(self.parameters()).device
        scores = []
        with torch.no_grad():
            doc_enc = tokenizer(document, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
            doc_ids = doc_enc["input_ids"].to(device)
            doc_mask = doc_enc["attention_mask"].to(device)
            for sentence in sentences:
                sent_enc = tokenizer(sentence, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
                score = self.forward(
                    sent_enc["input_ids"].to(device),
                    sent_enc["attention_mask"].to(device),
                    doc_ids,
                    doc_mask,
                )
                scores.append(score.item())
        return scores
