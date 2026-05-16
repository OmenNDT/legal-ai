import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

from src.data_pipeline.ingestion import DocumentParser, RawDocument
from src.data_pipeline.cleaning import TextCleaner
from src.data_pipeline.structuring import DocumentStructurer
from src.data_pipeline.chunking import DocumentChunker
from src.data_pipeline.tagging import MetadataTagger
from src.data_pipeline.embedding import EmbeddingGenerator
from src.data_pipeline.vector_storage import VectorStorageWriter
from src.data_pipeline.linking import EmbeddingLinker
from src.data_pipeline.index_builder import IndexAndKGBuilder

@dataclass
class StreamingConfig:
    watch_dir: str = field(default_factory=lambda: os.getenv("WATCH_DIR", "/home/minht/legal-ai/data_raw"))
    checkpoint_dir: str = field(default_factory=lambda: os.getenv("CHECKPOINT_DIR", "/home/minht/legal-ai/data_raw/.checkpoint"))
    trigger_interval: str = "10 seconds"
    db_config_path: Optional[str] = None
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"))
    chroma_host: str = field(default_factory=lambda: os.getenv("CHROMA_HOST", "localhost"))
    chroma_port: int = field(default_factory=lambda: int(os.getenv("CHROMA_PORT", "8001")))
    chroma_collection: str = field(default_factory=lambda: os.getenv("CHROMA_COLLECTION", "legal_chunks"))
    kg_path: str = "data/processed/knowledge_graph.gpickle"
    index_path: str = "data/processed/search_index.json"
    segmenter_dir: str = "./vncorenlp"
    spark_master: str = field(default_factory=lambda: os.getenv("SPARK_MASTER", "local[*]"))
    rebuild_indexes_per_batch: bool = False

class LegalStreamingJob:
    def __init__(self, config: StreamingConfig):
        self._config = config
        self._spark = self._init_spark()
        self._db = self._init_db()
        self._vector_store = self._init_vector_store()
        self._parser = DocumentParser()
        self._cleaner = TextCleaner(segmenter_dir = config.segmenter_dir)
        self._structurer = DocumentStructurer(db = self._db)
        self._chunker = DocumentChunker(db = self._db)
        self._tagger = MetadataTagger(db = self._db)
        self._embedder = EmbeddingGenerator(model_name = config.embedding_model)
        self._vector_writer = VectorStorageWriter(vector_store = self._vector_store)
        self._linker = EmbeddingLinker(db = self._db)
        self._index_builder = IndexAndKGBuilder(
            db=self._db,
            kg_path=config.kg_path,
            index_path=config.index_path,
        )

    def start(self) -> None:
        watch_path = Path(self._config.watch_dir).resolve()
        watch_path.mkdir(parents=True, exist_ok=True)
        old_dir = watch_path / "old"
        old_dir.mkdir(exist_ok = True)
        logger.info("[Streaming] Watching: %s", watch_path)

        stream_df = (
            self._spark.readStream
            .format("binaryFile")
            .option("pathGlobFilter", "*.pdf")
            .option("recursiveFileLookup", "false")
            .load("file://" + str(watch_path))
        )

        query = (
            stream_df.writeStream
            .foreachBatch(self._process_batch)
            .option("checkpointLocation", "file://" + str(Path(self._config.checkpoint_dir).resolve()))
            .trigger(processingTime=self._config.trigger_interval)
            .start()
        )

        query.awaitTermination()

    def _process_batch(self, batch_df, batch_id: int) -> None:
        rows = batch_df.collect()
        logger.info("[Batch %d] %d file(s) received", batch_id, len(rows))
        if not rows:
            return

        raw_docs, processed_paths = self._parse_rows(rows)
        logger.info("[Batch %d] Parsed: %d doc(s)", batch_id, len(raw_docs))
        if not raw_docs:
            return

        cleaned = self._cleaner.clean(raw_docs)
        logger.info("[Batch %d] Cleaned: %d doc(s)", batch_id, len(cleaned))
        structured = self._structurer.structure(cleaned)
        logger.info("[Batch %d] Structured: %d doc(s)", batch_id, len(structured))
        if not structured:
            return

        chunks = self._chunker.chunk(structured)
        logger.info("[Batch %d] Chunks: %d", batch_id, len(chunks))
        if not chunks:
            return

        self._tagger.tag(chunks)
        logger.info("[Batch %d] Tagged", batch_id)
        chunks_with_emb = self._embedder.generate(chunks)
        logger.info("[Batch %d] Embeddings generated", batch_id)
        self._vector_writer.write(chunks_with_emb)
        logger.info("[Batch %d] Written to ChromaDB", batch_id)
        self._linker.link([c["chunk_id"] for c in chunks_with_emb])
        logger.info("[Batch %d] Linked in Postgres", batch_id)

        if self._config.rebuild_indexes_per_batch:
            self._index_builder.build()
            logger.info("[Batch %d] Indexes rebuilt", batch_id)

        self._archive_files(processed_paths, batch_id)

    def _parse_rows(self, rows) -> tuple[list, list]:
        docs = []
        processed_paths = []
        for row in rows:
            filename = Path(row.path).name
            try:
                text = self._parser.parse_pdf(row.content)
                if text.strip():
                    docs.append(RawDocument(
                        filename=filename,
                        source_path=row.path,
                        raw_text=text,
                        file_format="pdf",
                    ))
                    processed_paths.append(row.path)
                    logger.info("[Parse] OK: %s (%d chars)", filename, len(text))
                else:
                    logger.warning("[Parse] Empty text: %s", filename)
            except Exception as e:
                logger.error("[Parse] FAILED: %s — %s", filename, e, exc_info=True)
        return docs, processed_paths

    def _archive_files(self, source_paths: list, batch_id: int) -> None:
        watch_path = Path(self._config.watch_dir).resolve()
        old_dir = watch_path / "old"
        for src_str in source_paths:
            src = Path(src_str.replace("file://", ""))
            if not src.exists():
                logger.warning("[Archive] Not found, skip: %s", src)
                continue
            dest = old_dir / src.name
            if dest.exists():
                dest = old_dir / f"{src.stem}_{batch_id}{src.suffix}"
            shutil.move(str(src), str(dest))
            logger.info("[Archive] Moved: %s → old/%s", src.name, dest.name)

    def stop(self) -> None:
        if self._db:
            self._db.disconnect()
        if self._spark:
            self._spark.stop()

    def _init_spark(self):
        from pyspark.sql import SparkSession
        return (
            SparkSession.builder
            .master(self._config.spark_master)
            .appName("LegalStreamingETL")
            .config("spark.executor.memory", "4g")
            .config("spark.driver.memory", "4g")
            .config("spark.sql.streaming.schemaInference", "true")
            .getOrCreate()
        )

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
        self.stop()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    from dotenv import load_dotenv
    load_dotenv()

    config = StreamingConfig()
    with LegalStreamingJob(config) as job:
        job.start()
