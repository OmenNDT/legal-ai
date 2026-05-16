"""Build search index and knowledge graph from legal corpus."""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.config import PROCESSED_DIR, SEARCH_INDEX_PATH, KG_PATH
from src.search.search_engine import LegalSearchEngine
from src.knowledge.graph_builder import LegalKnowledgeGraph
from src.common.text_processor import clean_vietnamese


def build_search_index(docs_path: str = None, output_path: str = None):
    """Build inverted index + BM25 + Trie from processed legal documents."""
    docs_path = Path(docs_path or PROCESSED_DIR / "legal_docs.json")
    output_path = output_path or str(SEARCH_INDEX_PATH)

    if not docs_path.exists():
        print(f"Error: {docs_path} not found. Run preprocess.py first.")
        return

    with open(docs_path, encoding="utf-8") as f:
        docs = json.load(f)

    print(f"Building search index from {len(docs)} documents...")
    engine = LegalSearchEngine()

    for i, doc in enumerate(docs):
        content = doc.get("content", doc.get("segmented_content", ""))
        if not content:
            content = doc.get("text", "")
        content = clean_vietnamese(content)

        metadata = {
            "title": doc.get("title", ""),
            "type": doc.get("type", ""),
            "law_number": doc.get("law_number", ""),
            "issuer": doc.get("issuer", ""),
            "content": content[:2000],  # Store truncated content for retrieval
        }
        if content:
            engine.add_document(content, metadata)
            if (i + 1) % 1000 == 0:
                print(f"  Indexed {i + 1}/{len(docs)} documents...")

    engine.build()
    print(f"  Index: {engine.index.vocabulary_size} terms, {engine.index.doc_count} documents")
    print(f"  Avg doc length: {engine.index.avg_doc_length:.1f} terms")

    # Save index
    engine.index.save(output_path)
    print(f"Search index saved → {output_path}")

    return engine


def build_knowledge_graph(docs_path: str = None, output_path: str = None):
    """Build knowledge graph from processed legal documents."""
    docs_path = Path(docs_path or PROCESSED_DIR / "legal_docs.json")
    output_path = output_path or str(KG_PATH)

    if not docs_path.exists():
        print(f"Error: {docs_path} not found. Run preprocess.py first.")
        return

    with open(docs_path, encoding="utf-8") as f:
        docs = json.load(f)

    print(f"Building knowledge graph from {len(docs)} documents...")
    kg = LegalKnowledgeGraph()

    for i, doc in enumerate(docs):
        kg.add_document(doc)
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(docs)} documents...")

    stats = kg.get_stats()
    print(f"  Nodes: {stats['nodes']}")
    print(f"  Edges: {stats['edges']}")
    print(f"  Node types: {stats['node_types']}")
    print(f"  Relation types: {stats['relation_types']}")

    kg.save(output_path)
    return kg


if __name__ == "__main__":
    print("=" * 60)
    print("Building search index...")
    print("=" * 60)
    build_search_index()

    print("\n" + "=" * 60)
    print("Building knowledge graph...")
    print("=" * 60)
    build_knowledge_graph()