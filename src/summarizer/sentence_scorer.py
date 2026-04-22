"""PhoBERT Sentence Scorer for extractive summarization.

Head 3 of the PhoBERT fine-tuning pipeline.
Architecture: PhoBERT encoder → mean pooling → concat[sent, doc, sent*doc] → Linear → score

Training: Binary classification (sentence in summary / not in summary)
Loss: BCEWithLogitsLoss
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class PhoBERTSentenceScorer(nn.Module):
    """Score sentence importance for extractive summarization.

    Given a sentence and its document context, predicts whether
    the sentence should be included in the summary.

    Architecture:
        1. Encode sentence and document with PhoBERT
        2. Mean pool each encoding
        3. Concatenate: [sent_emb, doc_emb, sent_emb * doc_emb]
        4. Feed through pre-classifier (768*3 → 768) + dropout + regression head

    Input dimension: 768 * 3 = 2304 (concat of sent, doc, element-wise product)
    Output dimension: 1 (importance score)
    """

    def __init__(self, model_name: str = "vinai/phobert-base", hidden_size: int = 768):
        super().__init__()
        self.phobert = AutoModel.from_pretrained(model_name)
        self.pre_classifier = nn.Linear(hidden_size * 3, hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.regression_head = nn.Linear(hidden_size, 1)

    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling over token embeddings, respecting attention mask."""
        token_embeddings = model_output.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, sent_ids, sent_mask, doc_ids, doc_mask):
        """Forward pass.

        Args:
            sent_ids: Token IDs for the sentence (batch, seq_len)
            sent_mask: Attention mask for the sentence (batch, seq_len)
            doc_ids: Token IDs for the document (batch, seq_len)
            doc_mask: Attention mask for the document (batch, seq_len)

        Returns:
            Tensor of shape (batch, 1) with raw importance scores
        """
        # Encode sentence and document
        sent_output = self.phobert(input_ids=sent_ids, attention_mask=sent_mask)
        doc_output = self.phobert(input_ids=doc_ids, attention_mask=doc_mask)

        # Mean pool
        sent_emb = self._mean_pooling(sent_output, sent_mask)
        doc_emb = self._mean_pooling(doc_output, doc_mask)

        # Combine: [sent, doc, sent * doc]
        combined = torch.cat([sent_emb, doc_emb, sent_emb * doc_emb], dim=1)

        # Feed through classifier
        x = torch.nn.ReLU()(self.pre_classifier(combined))
        x = self.dropout(x)
        score = self.regression_head(x)

        return score

    def score_sentences(self, sentences: list[str], document: str, tokenizer, max_length: int = 256):
        """Score all sentences in a document for summary inclusion.

        Args:
            sentences: List of sentences to score
            document: Full document text (for context)
            tokenizer: PhoBERT tokenizer
            max_length: Max token length

        Returns:
            List of float scores, one per sentence
        """
        self.eval()
        scores = []

        with torch.no_grad():
            # Encode document once
            doc_enc = tokenizer(
                document, max_length=max_length, truncation=True,
                padding="max_length", return_tensors="pt"
            )
            doc_ids = doc_enc["input_ids"].to(next(self.parameters()).device)
            doc_mask = doc_enc["attention_mask"].to(next(self.parameters()).device)

            for sentence in sentences:
                sent_enc = tokenizer(
                    sentence, max_length=max_length, truncation=True,
                    padding="max_length", return_tensors="pt"
                )
                sent_ids = sent_enc["input_ids"].to(next(self.parameters()).device)
                sent_mask = sent_enc["attention_mask"].to(next(self.parameters()).device)

                score = self.forward(sent_ids, sent_mask, doc_ids, doc_mask)
                scores.append(score.item())

        return scores