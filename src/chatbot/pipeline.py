"""Chatbot pipeline: intent → NER → dispatch to search/summarizer/KG → response.

The central orchestrator that connects all 4 modules:
1. Intent classification (PhoBERT Head 1)
2. Named Entity Recognition (PhoBERT Head 2)
3. Search (InvertedIndex + BM25 + Trie)
4. Summarization (PhoBERT Head 3 + TextRank)
5. Knowledge Graph reasoning (NetworkX)
"""

from dataclasses import dataclass
from typing import Optional

from src.common.config import INTENT_LABELS
from src.chatbot.intent_classifier import PhoBERTIntentClassifier
from src.chatbot.ner_tagger import PhoBERTNERTagger
from src.search.search_engine import LegalSearchEngine
from src.summarizer.summarizer import LegalSummarizer
from src.knowledge.graph_builder import LegalKnowledgeGraph
from src.knowledge.reasoner import LegalReasoner


@dataclass
class ChatResponse:
    """Structured response from the chatbot."""
    answer: str
    intent: str
    confidence: float
    entities: list[dict]
    sources: list[dict]
    reasoning: list[str]
    summary: Optional[str] = None


class LegalChatbot:
    """Legal Q&A chatbot orchestrating all modules.

    Flow: question → intent + NER → search → summarize → KG → response
    """

    # Intent to module routing
    INTENT_ROUTING = {
        "hoi_dieu_khoan": ("search", "summarize"),
        "hoi_luat_moi": ("search", "summarize"),
        "tra_cuu_thong_tu": ("search", "summarize", "kg"),
        "tra_cuu_nghi_dinh": ("search", "summarize", "kg"),
        "tra_cuu_luat": ("search", "summarize", "kg"),
        "hoi_hieu_luc": ("kg",),
        "hoi_sua_doi": ("kg", "search"),
        "hoi_dinh_nghia": ("search", "summarize"),
        "hoi_thu_tuc": ("search", "summarize"),
        "hoi_hinh_phat": ("search", "summarize", "kg"),
        "hoi_quyen_loi": ("search", "summarize"),
        "hoi_nghia_vu": ("search", "summarize"),
        "hoi_bieu_phi": ("search", "summarize"),
        "hoi_thoi_han": ("search", "kg"),
        "hoi_doi_tuong": ("search", "summarize"),
        "hoi_dieu_kien": ("search", "summarize"),
        "hoi_phan_biet": ("search", "summarize"),
        "hoi_so_sanh": ("search", "kg", "summarize"),
        "hoi_tom_tat": ("search", "summarize"),
        "hoi_tong_hop": ("search", "kg", "summarize"),
    }

    def __init__(
        self,
        intent_model_path: Optional[str] = None,
        ner_model_path: Optional[str] = None,
        scorer_model_path: Optional[str] = None,
        lora_checkpoint_path: Optional[str] = None,
        search_engine: Optional[LegalSearchEngine] = None,
        knowledge_graph: Optional[LegalKnowledgeGraph] = None,
    ):
        # Load models lazily
        self._intent_classifier = None
        self._ner_tagger = None
        self._summarizer = None
        self._search_engine = search_engine
        self._kg = knowledge_graph
        self._reasoner = None

        self._intent_model_path = intent_model_path
        self._ner_model_path = ner_model_path
        self._scorer_model_path = scorer_model_path
        self._lora_checkpoint_path = lora_checkpoint_path
        self._lora_loaded = False

    def _load_lora(self):
        """Load LoRA checkpoint for both intent and NER."""
        if self._lora_loaded or not self._lora_checkpoint_path:
            return
        import torch
        from transformers import AutoTokenizer

        checkpoint = torch.load(self._lora_checkpoint_path, map_location="cpu")
        phobert_state = {k.replace("phobert.", ""): v
                         for k, v in checkpoint["model_state_dict"].items()
                         if k.startswith("phobert.")}

        # Intent
        self._intent_classifier = PhoBERTIntentClassifier()
        self._intent_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        intent_state = {k.replace("intent_classifier.", ""): v
                        for k, v in checkpoint["model_state_dict"].items()
                        if k.startswith("intent_classifier.")}
        self._intent_classifier.phobert.load_state_dict(phobert_state, strict=False)
        self._intent_classifier.classifier.load_state_dict(intent_state, strict=False)
        self._intent_classifier.eval()

        # NER
        self._ner_tagger = PhoBERTNERTagger()
        self._ner_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        ner_state = {k.replace("ner_classifier.", ""): v
                     for k, v in checkpoint["model_state_dict"].items()
                     if k.startswith("ner_classifier.")}
        self._ner_tagger.phobert.load_state_dict(phobert_state, strict=False)
        self._ner_tagger.classifier.load_state_dict(ner_state, strict=False)
        self._ner_tagger.eval()

        self._lora_loaded = True

    def _load_intent(self):
        if self._intent_classifier is None:
            if self._lora_checkpoint_path:
                self._load_lora()
            else:
                from transformers import AutoTokenizer
                self._intent_classifier = PhoBERTIntentClassifier()
                self._intent_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
                if self._intent_model_path:
                    import torch
                    state_dict = torch.load(self._intent_model_path, map_location="cpu")
                    self._intent_classifier.load_state_dict(state_dict)
                self._intent_classifier.eval()

    def _load_ner(self):
        if self._ner_tagger is None:
            if self._lora_checkpoint_path:
                self._load_lora()
            else:
                from transformers import AutoTokenizer
                self._ner_tagger = PhoBERTNERTagger()
                self._ner_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
                if self._ner_model_path:
                    import torch
                    state_dict = torch.load(self._ner_model_path, map_location="cpu")
                    self._ner_tagger.load_state_dict(state_dict)
                self._ner_tagger.eval()

    def _load_summarizer(self):
        if self._summarizer is None:
            self._summarizer = LegalSummarizer(model_path=self._scorer_model_path)

    def _load_reasoner(self):
        if self._reasoner is None and self._kg is not None:
            self._reasoner = LegalReasoner(self._kg)

    def ask(self, question: str, top_k_docs: int = 5, top_k_sentences: int = 4) -> ChatResponse:
        """Process a legal question through the full pipeline.

        Args:
            question: Vietnamese legal question
            top_k_docs: Number of documents to retrieve from search
            top_k_sentences: Number of sentences for summary

        Returns:
            ChatResponse with answer, sources, reasoning
        """
        # Step 1: Intent classification
        self._load_intent()
        intent_result = self._intent_classifier.predict(question, self._intent_tokenizer)
        intent = intent_result["intent"]
        confidence = intent_result["confidence"]

        # Step 2: Named Entity Recognition
        self._load_ner()
        entities = self._ner_tagger.predict(question, self._ner_tokenizer)

        # Step 3: Route to modules based on intent
        modules = self.INTENT_ROUTING.get(intent, ("search", "summarize"))
        sources = []
        reasoning = []
        summary = None

        # Step 3a: Search (if needed)
        if "search" in modules and self._search_engine is not None:
            search_results = self._search_engine.search(question, top_k=top_k_docs)
            sources = search_results

        # Step 3b: Summarize (if needed and search results exist)
        if "summarize" in modules and sources:
            self._load_summarizer()
            # Use top search result as document for summarization
            top_doc = sources[0] if sources else None
            if top_doc and "content" in top_doc:
                summary_result = self._summarizer.summarize(
                    document=top_doc["content"],
                    query=question,
                    top_k=top_k_sentences,
                )
                summary = summary_result["summary"]
                reasoning.append(f"Tóm tắt từ {top_doc.get('title', 'văn bản')}")

        # Step 3c: Knowledge Graph (if needed)
        if "kg" in modules and self._reasoner is not None:
            # Extract document IDs from entities
            for entity in entities:
                if entity["entity"] in ("THONG_TU", "NGHI_DINH", "LUAT"):
                    doc_id = f"{entity['entity']}_{entity['text'].replace(' ', '_')}"
                    kg_result = self._reasoner.check_validity(doc_id)
                    reasoning.extend(kg_result.reasoning_steps)

        # Step 4: Build response
        answer = self._build_answer(intent, entities, sources, summary, reasoning)

        return ChatResponse(
            answer=answer,
            intent=intent,
            confidence=confidence,
            entities=entities,
            sources=sources,
            reasoning=reasoning,
            summary=summary,
        )

    def _build_answer(self, intent, entities, sources, summary, reasoning):
        """Build a natural language answer from module outputs."""
        parts = []

        if summary:
            parts.append(summary)

        if reasoning:
            parts.append("\n".join(reasoning))

        if sources:
            source_texts = [f"- {s.get('title', s.get('doc_id', 'N/A'))}" for s in sources[:3]]
            parts.append("Nguồn tham khảo:\n" + "\n".join(source_texts))

        if not parts:
            parts.append("Xin lỗi, tôi không tìm thấy thông tin liên quan. Vui lòng thử lại với câu hỏi cụ thể hơn.")

        return "\n\n".join(parts)