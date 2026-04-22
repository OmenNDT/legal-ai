"""Legal text summarizer combining PhoBERT relevance + TextRank centrality + position.

The complete extractive summarization pipeline:
1. Split document into sentences
2. Encode sentences with PhoBERT
3. Score each sentence: relevance(query, sent) + centrality(TextRank) + position
4. Select top-K sentences by combined score
5. Reorder by original position and return summary
"""

from src.common.config import SUMMARY_WEIGHTS, SEARCH_TOP_K
from src.common.text_processor import segment_sentences, segment_text
from src.summarizer.textrank import (
    build_similarity_matrix, textrank, compute_position_scores, select_sentences,
)
from src.summarizer.sentence_scorer import PhoBERTSentenceScorer


class LegalSummarizer:
    """Extractive summarizer for Vietnamese legal documents.

    Uses PhoBERT for relevance scoring + TextRank for centrality + position bonus.
    """

    def __init__(self, model_path: str = None, model_name: str = "vinai/phobert-base"):
        self.tokenizer = None
        self.scorer = None
        self.model_name = model_name
        self.model_path = model_path
        self.weights = SUMMARY_WEIGHTS

    def _load_model(self):
        """Lazy-load model and tokenizer."""
        if self.scorer is not None:
            return

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.scorer = PhoBERTSentenceScorer(self.model_name)
        if self.model_path:
            import torch
            state_dict = torch.load(self.model_path, map_location="cpu")
            self.scorer.load_state_dict(state_dict)
        self.scorer.eval()

    def summarize(
        self,
        document: str,
        query: str = "",
        top_k: int = 4,
        min_gap: int = 0,
        use_model: bool = True,
    ) -> dict:
        """Summarize a legal document.

        Args:
            document: Full document text
            query: User question (for relevance scoring). If empty, uses centrality only.
            top_k: Number of sentences to include in summary
            min_gap: Minimum gap between selected sentences (0 = no constraint)
            use_model: Whether to use PhoBERT scorer (False = TextRank only)

        Returns:
            Dict with 'summary', 'scores', 'selected_indices', 'sentences'
        """
        # Step 1: Split into sentences
        sentences = segment_sentences(document)
        if len(sentences) <= top_k:
            return {
                "summary": " ".join(sentences),
                "sentences": sentences,
                "selected_indices": list(range(len(sentences))),
                "scores": {"relevance": [1.0] * len(sentences),
                          "centrality": [1.0] * len(sentences),
                          "position": [1.0] * len(sentences)},
            }

        # Step 2: Score sentences
        relevance_scores = self._compute_relevance(sentences, query, use_model)
        centrality_scores = self._compute_centrality(sentences, use_model)
        position_scores = compute_position_scores(len(sentences))

        # Step 3: Select top-K
        selected = select_sentences(
            relevance_scores, centrality_scores, position_scores,
            weights=self.weights, top_k=top_k, min_gap=min_gap,
        )

        # Step 4: Build summary (in original order)
        summary = " ".join(sentences[i] for i in selected)

        return {
            "summary": summary,
            "sentences": sentences,
            "selected_indices": selected,
            "scores": {
                "relevance": relevance_scores.tolist() if hasattr(relevance_scores, 'tolist') else list(relevance_scores),
                "centrality": centrality_scores.tolist() if hasattr(centrality_scores, 'tolist') else list(centrality_scores),
                "position": position_scores.tolist() if hasattr(position_scores, 'tolist') else list(position_scores),
            },
        }

    def _compute_relevance(self, sentences: list[str], query: str, use_model: bool):
        """Compute relevance scores for each sentence relative to query."""
        import numpy as np

        if not query:
            return np.ones(len(sentences))

        if use_model:
            try:
                self._load_model()
                model_scores = self.scorer.score_sentences(sentences, query, self.tokenizer)
                return np.array(model_scores)
            except Exception:
                pass  # Fall back to TF-IDF cosine

        # Fallback: TF-IDF cosine similarity
        from sklearn.feature_extraction.text import TfidfVectorizer
        all_texts = [query] + sentences
        vectorizer = TfidfVectorizer(max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        query_vec = tfidf_matrix[0]
        sent_vecs = tfidf_matrix[1:]

        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_vec, sent_vecs).flatten()
        return similarities

    def _compute_centrality(self, sentences: list[str], use_model: bool):
        """Compute TextRank centrality scores."""
        import numpy as np

        if use_model:
            try:
                self._load_model()
                # Use PhoBERT embeddings for similarity
                embeddings = []
                for sent in sentences:
                    enc = self.tokenizer(sent, max_length=256, truncation=True,
                                        padding="max_length", return_tensors="pt")
                    with __import__('torch').no_grad():
                        output = self.scorer.phobert(input_ids=enc["input_ids"],
                                                     attention_mask=enc["attention_mask"])
                    emb = self.scorer._mean_pooling(output, enc["attention_mask"])
                    embeddings.append(emb.squeeze().numpy())
            except Exception:
                embeddings = None
        else:
            embeddings = None

        if embeddings is None or len(embeddings[0]) == 0:
            # Fallback: TF-IDF embeddings for TextRank
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=5000)
            tfidf_matrix = vectorizer.fit_transform(sentences)
            embeddings = tfidf_matrix.toarray()

        sim_matrix = build_similarity_matrix(embeddings if isinstance(embeddings, list) and isinstance(embeddings[0], np.ndarray) else [e.flatten() for e in embeddings])
        scores = textrank(sim_matrix)
        return scores