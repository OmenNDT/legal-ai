import os
from dataclasses import dataclass, field
from typing import Optional

from src.data_pipeline.ingestion import DocumentIngester
from src.data_pipeline.cleaning import TextCleaner
from src.data_pipeline.structuring import DocumentStructurer
from src.data_pipeline.chunking import DocumentChunker
from src.data_pipeline.tagging import MetadataTagger
from src.data_pipeline.embedding import EmbeddingGenerator
from src.data_pipeline.vector_storage import VectorStorageWriter
from src.data_pipeline.linking import EmbeddingLinker
from src.data_pipeline.index_builder import IndexAndKGBuilder
from src.data_pipeline.validation import PipelineValidator, ValidationResult

@dataclass
class SparkETLConfig:
    source_path: str
    db_config_path: Optional[str] = None
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"))
    chroma_host: str = field(default_factory=lambda: os.getenv("CHROMA_HOST", "localhost"))
    chroma_port: int = field(default_factory=lambda: int(os.getenv("CHROMA_PORT", "8001")))
    chroma_collection: str = field(default_factory=lambda: os.getenv("CHROMA_COLLECTION", "legal_chunks"))
    kg_path: str = "data/processed/knowledge_graph.gpickle"
    index_path: str = "data/processed/search_index.json"
    segmenter_dir: str = "./vncorenlp"
    spark_master: str = field(default_factory=lambda: os.getenv("SPARK_MASTER", "local[*]"))
    spark_app_name: str = "LegalETL"

class SparkETLPipeline:
    def __init__(self, config: SparkETLConfig):
        self._config = config
        self._spark = self._init_spark()
        self._db = self._init_db()
        self._vector_store = self._init_vector_store()

        self._ingester = DocumentIngester(spark=self._spark)
        self._cleaner = TextCleaner(spark=self._spark, segmenter_dir=config.segmenter_dir)
        self._structurer = DocumentStructurer(db=self._db)
        self._chunker = DocumentChunker(db=self._db)
        self._tagger = MetadataTagger(db=self._db)
        self._embedder = EmbeddingGenerator(model_name=config.embedding_model)
        self._vector_writer = VectorStorageWriter(vector_store=self._vector_store)
        self._linker = EmbeddingLinker(db=self._db)
        self._index_builder = IndexAndKGBuilder(
            db = self._db,
            kg_path = config.kg_path,
            index_path = config.index_path,
        )
        self._validator = PipelineValidator(db=self._db, vector_store=self._vector_store)

    def run(self) -> ValidationResult:
        raw_docs = self._ingester.ingest(self._config.source_path)
        cleaned_docs = self._cleaner.clean(raw_docs)
        structured_docs = self._structurer.structure(cleaned_docs)
        chunks = self._chunker.chunk(structured_docs)
        self._tagger.tag(chunks)
        chunks_with_embeddings = self._embedder.generate(chunks)
        embedding_ids = self._vector_writer.write(chunks_with_embeddings)
        linked_chunk_ids = [c["chunk_id"] for c in chunks_with_embeddings]
        self._linker.link(linked_chunk_ids)
        self._index_builder.build()
        return self._validator.validate()

    def _init_spark(self):
        try:
            from pyspark.sql import SparkSession
            return (
                SparkSession.builder
                .master(self._config.spark_master)
                .appName(self._config.spark_app_name)
                .config("spark.executor.memory", "4g")
                .config("spark.driver.memory", "4g")
                .config("spark.sql.execution.arrow.pyspark.enabled", "true")
                .getOrCreate()
            )
        except ImportError:
            return None

    def _init_db(self):
        from src.config.database_connection import DatabaseConnection
        db = DatabaseConnection(self._config.db_config_path)
        db.connect()
        return db

    def _init_vector_store(self):
        from src.search.vector_store import VectorStore
        return VectorStore(
            collection_name=self._config.chroma_collection,
            embedding_model=self._config.embedding_model,
            host=self._config.chroma_host,
            port=self._config.chroma_port,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._db:
            self._db.disconnect()
        if self._spark:
            self._spark.stop()
