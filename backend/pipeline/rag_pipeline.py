import os
from dataclasses import dataclass, field
from typing import Optional
import json

from backend.pipeline.query_understanding import QueryUnderstanding
from backend.pipeline.retrieval import Retrieval
from backend.pipeline.reranking import ReRanking
from backend.pipeline.reasoning import Reasoning
from backend.pipeline.post_processing import PostProcessing, FinalResponse

@dataclass
class PipelineConfig:
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    llm_base_url: Optional[str] = field(default_factory=lambda: os.getenv("LLM_BASE_URL") or None)
    chroma_host: str = field(default_factory=lambda: os.getenv("CHROMA_HOST", "localhost"))
    chroma_port: int = field(default_factory=lambda: int(os.getenv("CHROMA_PORT", "8001")))
    chroma_collection: str = field(default_factory=lambda: os.getenv("CHROMA_COLLECTION", "legal_chunks"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"))
    cross_encoder_model: str = field(default_factory=lambda: os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    intent_model_path: Optional[str] = None
    ner_model_path: Optional[str] = None
    use_postgres_bm25: bool = True
    db_config_path: Optional[str] = None
    kg_path: Optional[str] = None


class RAGPipeline:
    def __init__(self, config: PipelineConfig):
        self._config = config
        self._db = self._init_db()
        self._llm = self._init_llm()
        self._hybrid_search = self._init_search()
        self._cross_encoder = self._init_cross_encoder()
        self._kg_reasoner = self._init_kg_reasoner()

        from backend.summarizer.summarizer import LegalSummarizer
        self._summarizer = LegalSummarizer()

        intent_clf, intent_tok, ner_tagger, ner_tok = self._init_nlp_models()

        self._query_understanding = QueryUnderstanding(
            intent_classifier=intent_clf,
            ner_tagger=ner_tagger,
            intent_tokenizer=intent_tok,
            ner_tokenizer=ner_tok,
            llm_client=self._llm,
        )
        self._retrieval = Retrieval(hybrid_search=self._hybrid_search, db=self._db)
        self._reranking = ReRanking(cross_encoder=self._cross_encoder, summarizer=self._summarizer)
        self._reasoning = Reasoning(llm_client=self._llm, kg_reasoner=self._kg_reasoner)
        self._post_processing = PostProcessing(db=self._db)

    def query(self, user_query: str) -> FinalResponse:
        structured_query = self._query_understanding.process(user_query)

        if self._db:
            query_id = self._save_query(structured_query)
            structured_query.query_id = query_id

        candidates = self._retrieval.retrieve(structured_query)
        prompt = self._reranking.process(candidates, structured_query)
        answer = self._reasoning.generate(prompt, structured_query)
        return self._post_processing.process(answer, candidates, structured_query)

    def _save_query(self, structured_query) -> Optional[int]:
        try:
            rows = self._db.execute_query(
                "INSERT INTO user_queries (query_text, normalized_query, intent, entities) "
                "VALUES (%s, %s, %s, %s) RETURNING id",
                (
                    structured_query.raw_query,
                    structured_query.normalized_query,
                    structured_query.intent,
                    json.dumps(structured_query.entities, ensure_ascii=False),
                ),
            )
            return rows[0][0] if rows else None
        except Exception:
            return None

    def _init_db(self):
        try:
            from config.database.database_connection import DatabaseConnection
            db = DatabaseConnection(self._config.db_config_path)
            db.connect()
            return db
        except Exception:
            return None

    def _init_llm(self):
        from backend.llm.client import LLMClient
        return LLMClient(
            provider=self._config.llm_provider,
            model=self._config.llm_model,
            api_key=self._config.llm_api_key,
            base_url=self._config.llm_base_url,
        )

    def _init_search(self):
        from backend.search.vector_store import VectorStore
        from backend.search.hybrid_search import HybridSearch

        vector_store = VectorStore(
            collection_name=self._config.chroma_collection,
            embedding_model=self._config.embedding_model,
            host=self._config.chroma_host,
            port=self._config.chroma_port,
        )

        if self._config.use_postgres_bm25 and self._db:
            from backend.search.postgres_bm25 import PostgresBM25
            bm25 = PostgresBM25(self._db)
        else:
            from backend.search.inverted_index import InvertedIndex
            from backend.search.bm25 import BM25
            bm25 = BM25(InvertedIndex())

        return HybridSearch(bm25=bm25, vector_store=vector_store)

    def _init_cross_encoder(self):
        try:
            from sentence_transformers import CrossEncoder
            return CrossEncoder(self._config.cross_encoder_model)
        except Exception:
            return None

    def _init_kg_reasoner(self):
        if not self._config.kg_path:
            return None
        try:
            from backend.knowledge.graph_builder import LegalKnowledgeGraph
            from backend.knowledge.reasoner import LegalReasoner
            kg = LegalKnowledgeGraph()
            kg.load(self._config.kg_path)
            return LegalReasoner(kg)
        except Exception:
            return None

    def _init_nlp_models(self):
        from transformers import AutoTokenizer
        from backend.chatbot.intent_classifier import PhoBERTIntentClassifier
        from backend.chatbot.ner_tagger import PhoBERTNERTagger

        shared_tok = AutoTokenizer.from_pretrained("vinai/phobert-base")

        intent_clf = PhoBERTIntentClassifier()
        if self._config.intent_model_path:
            import torch
            intent_clf.load_state_dict(torch.load(self._config.intent_model_path, map_location="cpu"))
        intent_clf.eval()

        ner_tagger = PhoBERTNERTagger()
        if self._config.ner_model_path:
            import torch
            ner_tagger.load_state_dict(torch.load(self._config.ner_model_path, map_location="cpu"))
        ner_tagger.eval()

        return intent_clf, shared_tok, ner_tagger, shared_tok
