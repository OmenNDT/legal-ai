"""Build legal knowledge graph from legal documents and QA data."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.config import PROCESSED_DIR, KG_PATH, QA_DATA_PATH
from src.knowledge.graph_builder import LegalKnowledgeGraph
from src.knowledge.entity_extractor import LegalEntityExtractor, LegalRelationExtractor
from src.common.text_processor import clean_vietnamese, extract_legal_references


def build_from_legal_docs(docs_path: str = None):
    """Build KG from processed legal documents."""
    docs_path = Path(docs_path or PROCESSED_DIR / "legal_docs.json")

    if not docs_path.exists():
        print(f"Error: {docs_path} not found. Run preprocess.py first.")
        return None

    with open(docs_path, encoding="utf-8") as f:
        docs = json.load(f)

    print(f"Building KG from {len(docs)} legal documents...")
    kg = LegalKnowledgeGraph()

    for doc in docs:
        kg.add_document(doc)

    stats = kg.get_stats()
    print(f"  Nodes: {stats['nodes']}, Edges: {stats['edges']}")
    print(f"  Node types: {stats['node_types']}")
    print(f"  Relation types: {stats['relation_types']}")

    return kg


def build_from_qa_data(qa_path: str = None):
    """Build KG entities and relations from QA data (supplementary)."""
    qa_path = Path(qa_path or QA_DATA_PATH)

    if not qa_path.exists():
        print(f"Error: {qa_path} not found.")
        return None

    with open(qa_path, encoding="utf-8") as f:
        data = json.load(f)

    # Handle different QA data formats
    if isinstance(data, dict) and "sample" in data:
        items = [_unwrap(data["sample"])]
    elif isinstance(data, list):
        items = data[:1000]  # Process first 1000 for KG
    else:
        print("Unexpected QA data format.")
        return None

    print(f"Extracting KG from {len(items)} QA pairs...")
    entity_extractor = LegalEntityExtractor()
    relation_extractor = LegalRelationExtractor()

    kg = LegalKnowledgeGraph()

    for item in items:
        context = item.get("context", "")
        question = item.get("question", "")
        if not context:
            continue

        # Extract entities from question and context
        q_entities = entity_extractor.extract(question)
        c_entities = entity_extractor.extract(context)

        # Create QA node
        qa_id = f"QA_{item.get('qid', hash(question) % 100000)}"
        from src.knowledge.entity_extractor import LegalEntity
        qa_entity = LegalEntity(
            id=qa_id,
            type="QUESTION",
            name=question[:100],
            properties={"full_question": question, "context_id": item.get("cid", "")},
        )
        kg.add_entity(qa_entity)

        # Add entities and link to question
        for entity in q_entities + c_entities:
            kg.add_entity(entity)
            kg.add_relation(LegalRelation(
                source_id=qa_id,
                target_id=entity.id,
                relation_type="THAM_CHIEU",
                properties={"evidence": entity.name},
            ))

    stats = kg.get_stats()
    print(f"  QA KG: {stats['nodes']} nodes, {stats['edges']} edges")
    return kg


def _unwrap(entry):
    return {
        "question": entry.get("question", ""),
        "context": entry.get("context_list", entry.get("context", "")),
        "qid": entry.get("qid", ""),
        "cid": entry.get("cid", ""),
    }


if __name__ == "__main__":
    print("Building Knowledge Graph from legal documents...")
    kg_docs = build_from_legal_docs()

    print("\nBuilding Knowledge Graph from QA data...")
    kg_qa = build_from_qa_data()

    # Merge if both exist
    if kg_docs and kg_qa:
        # Add QA nodes and edges to docs KG
        for node, data in kg_qa.G.nodes(data=True):
            kg_docs.G.add_node(node, **data)
        for source, target, data in kg_qa.G.edges(data=True):
            kg_docs.G.add_edge(source, target, **data)

        stats = kg_docs.get_stats()
        print(f"\nMerged KG: {stats['nodes']} nodes, {stats['edges']} edges")

    if kg_docs:
        kg_docs.save()