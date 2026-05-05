from src.common.config import SUMMARY_WEIGHTS
from src.common.text_processor import segment_sentences
from src.summarizer.textrank import (
    build_similarity_matrix, textrank, compute_position_scores, select_sentences,
)
from src.summarizer.sentence_scorer import PhoBERTSentenceScorer

class LegalSummarizer:
    def __init__(self, model_path: str = None, model_name: str = "vinai/phobert-base"):
        self.tokenizer = None
        self.scorer = None
        self.model_name = model_name
        self.model_path = model_path
        self.weights = SUMMARY_WEIGHTS

    def _load_model(self):
        if self.scorer is not None:
            return
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.scorer = PhoBERTSentenceScorer(self.model_name)
        if self.model_path:
            import torch
            self.scorer.load_state_dict(torch.load(self.model_path, map_location="cpu"))
        self.scorer.eval()

    def summarize(self, document: str, query: str = "", top_k: int = 4, min_gap: int = 0, use_model: bool = True) -> dict:
        sentences = segment_sentences(document)
        if len(sentences) <= top_k:
            return {
                "summary": " ".join(sentences),
                "sentences": sentences,
                "selected_indices": list(range(len(sentences))),
                "scores": {
                    "relevance": [1.0] * len(sentences),
                    "centrality": [1.0] * len(sentences),
                    "position": [1.0] * len(sentences),
                },
            }
        relevance_scores = self._compute_relevance(sentences, query, use_model)
        centrality_scores = self._compute_centrality(sentences, use_model)
        position_scores = compute_position_scores(len(sentences))
        selected = select_sentences(
            relevance_scores, centrality_scores, position_scores,
            weights=self.weights, top_k=top_k, min_gap=min_gap,
        )
        summary = " ".join(sentences[i] for i in selected)
        return {
            "summary": summary,
            "sentences": sentences,
            "selected_indices": selected,
            "scores": {
                "relevance": relevance_scores.tolist() if hasattr(relevance_scores, "tolist") else list(relevance_scores),
                "centrality": centrality_scores.tolist() if hasattr(centrality_scores, "tolist") else list(centrality_scores),
                "position": position_scores.tolist() if hasattr(position_scores, "tolist") else list(position_scores),
            },
        }

    def _compute_relevance(self, sentences: list, query: str, use_model: bool):
        import numpy as np
        if not query:
            return np.ones(len(sentences))
        if use_model:
            try:
                self._load_model()
                return np.array(self.scorer.score_sentences(sentences, query, self.tokenizer))
            except Exception:
                pass
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vectorizer = TfidfVectorizer(max_features=5000)
        tfidf_matrix = vectorizer.fit_transform([query] + sentences)
        return cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:]).flatten()

    def _compute_centrality(self, sentences: list, use_model: bool):
        import numpy as np
        embeddings = None
        if use_model:
            try:
                self._load_model()
                embs = []
                import torch
                for sent in sentences:
                    enc = self.tokenizer(sent, max_length=256, truncation=True, padding="max_length", return_tensors="pt")
                    with torch.no_grad():
                        out = self.scorer.phobert(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
                    emb = self.scorer._mean_pooling(out, enc["attention_mask"])
                    embs.append(emb.squeeze().numpy())
                embeddings = embs
            except Exception:
                embeddings = None
        if embeddings is None:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=5000)
            embeddings = vectorizer.fit_transform(sentences).toarray().tolist()
        sim_matrix = build_similarity_matrix([np.array(e) for e in embeddings])
        return textrank(sim_matrix)
