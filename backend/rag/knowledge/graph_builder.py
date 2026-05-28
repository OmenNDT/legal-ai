"""Knowledge Graph construction from Vietnamese legal documents.

Builds a NetworkX DiGraph with:
- 9 entity types as nodes
- 7 relation types as edges
- Properties: effective_date, status, etc.
"""

import json
import pickle
from pathlib import Path
from typing import Optional

import networkx as nx

from backend.common.config import (
    LEGAL_DATA_PATH, PROCESSED_DIR, KG_PATH,
    ENTITY_TYPES, RELATION_TYPES, DOCUMENT_TYPES,
)
from backend.common.text_processor import clean_vietnamese, segment_sentences
from backend.knowledge.entity_extractor import (
    LegalEntity, LegalRelation,
    LegalEntityExtractor, LegalRelationExtractor,
    determine_document_type,
)


class LegalKnowledgeGraph:
    """Legal knowledge graph using NetworkX DiGraph."""

    # Node colors for visualization
    NODE_COLORS = {
        "THONG_TU": "#4CAF50",
        "NGHI_DINH": "#2196F3",
        "LUAT": "#FF5722",
        "BO_LUAT": "#9C27B0",
        "HIEN_PHAP": "#F44336",
        "DIEU": "#FF9800",
        "KHOAN": "#FFC107",
        "DIEM": "#8BC34A",
        "CO_QUAN": "#607D8B",
        "KHAISUAT": "#795548",
        "NGAY_THANG": "#009688",
    }

    # Edge colors for visualization
    EDGE_COLORS = {
        "DUA_TREN": "#2196F3",
        "THAM_CHIEU": "#4CAF50",
        "HET_HIEU_LUC": "#F44336",
        "THAY_THE": "#FF9800",
        "SUA_DOI_BO_SUNG": "#9C27B0",
        "HUONG_DAN": "#00BCD4",
        "CHUA": "#607D8B",
    }

    def __init__(self):
        self.G = nx.DiGraph()
        self._entity_extractor = LegalEntityExtractor()
        self._relation_extractor = LegalRelationExtractor()

    @property
    def node_count(self) -> int:
        return self.G.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self.G.number_of_edges()

    def add_entity(self, entity: LegalEntity):
        """Add a legal entity as a node."""
        self.G.add_node(
            entity.id,
            type=entity.type,
            name=entity.name,
            **entity.properties,
        )

    def add_relation(self, relation: LegalRelation):
        """Add a legal relation as an edge."""
        # Ensure both nodes exist
        if relation.source_id not in self.G:
            self.G.add_node(relation.source_id, type="UNKNOWN", name=relation.source_id)
        if relation.target_id not in self.G:
            self.G.add_node(relation.target_id, type="UNKNOWN", name=relation.target_id)

        self.G.add_edge(
            relation.source_id,
            relation.target_id,
            relation_type=relation.relation_type,
            **relation.properties,
        )

    def add_document(self, doc: dict):
        """Process a legal document and add all entities and relations to the graph.

        Args:
            doc: Document dict with keys: id, title, type, content, etc.
        """
        content = doc.get("content", doc.get("text", ""))
        title = doc.get("title", "")
        doc_type = doc.get("type", "") or determine_document_type(title, content)

        # Create document entity
        doc_id = doc.get("id", title.replace(" ", "_")[:50])
        doc_entity = LegalEntity(
            id=doc_id,
            type=doc_type.upper() if doc_type else "OTHER",
            name=title or doc_id,
            properties={
                "effective_date": doc.get("effective_date", ""),
                "issuer": doc.get("issuer", ""),
                "law_number": doc.get("law_number", ""),
                "status": doc.get("status", "active"),
            },
        )
        self.add_entity(doc_entity)

        # Extract entities
        entities = self._entity_extractor.extract(content)
        for entity in entities:
            self.add_entity(entity)

        # Extract relations
        relations = self._relation_extractor.extract(content, doc_entity)
        for relation in relations:
            self.add_relation(relation)

    def build_from_corpus(self, docs: list[dict]):
        """Build graph from a list of legal documents."""
        for doc in docs:
            self.add_document(doc)

    def is_valid(self, doc_id: str, as_of: Optional[str] = None) -> dict:
        """Check if a document is still in force.

        Args:
            doc_id: Document node ID
            as_of: Date string (optional, defaults to current)

        Returns:
            Dict with 'valid' (bool), 'status' (str), 'amended_by' (list)
        """
        if doc_id not in self.G:
            return {"valid": False, "status": "not_found", "amended_by": []}

        node = self.G.nodes[doc_id]
        status = node.get("status", "unknown")

        # Check for HET_HIEU_LUC edges pointing to this document
        expired_by = []
        for source, target, data in self.G.in_edges(doc_id, data=True):
            if data.get("relation_type") == "HET_HIEU_LUC":
                expired_by.append({"source": source, "evidence": data.get("evidence", "")})

        # Check for SUA_DOI_BO_SUNG edges
        amended_by = []
        for source, target, data in self.G.in_edges(doc_id, data=True):
            if data.get("relation_type") == "SUA_DOI_BO_SUNG":
                amended_by.append({"source": source, "evidence": data.get("evidence", "")})

        # Check for THAY_THE edges
        replaced_by = []
        for source, target, data in self.G.in_edges(doc_id, data=True):
            if data.get("relation_type") == "THAY_THE":
                replaced_by.append({"source": source, "evidence": data.get("evidence", "")})

        is_valid = status != "expired" and len(expired_by) == 0 and len(replaced_by) == 0

        return {
            "valid": is_valid,
            "status": status,
            "expired_by": expired_by,
            "amended_by": amended_by,
            "replaced_by": replaced_by,
        }

    def amendment_chain(self, doc_id: str) -> list[dict]:
        """Trace the full amendment chain for a document.

        Returns list of amendments in chronological order.
        """
        chain = []
        visited = set()
        queue = [doc_id]

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            for source, target, data in self.G.in_edges(current, data=True):
                if data.get("relation_type") in ("SUA_DOI_BO_SUNG", "THAY_THE"):
                    chain.append({
                        "amending_doc": source,
                        "relation": data["relation_type"],
                        "evidence": data.get("evidence", ""),
                        "amended_doc": current,
                    })
                    queue.append(source)

        return chain

    def related_docs(self, doc_id: str, max_depth: int = 2) -> list[dict]:
        """Find all documents related to a given document within max_depth hops."""
        if doc_id not in self.G:
            return []

        related = []
        for source, target in nx.bfs_edges(self.G, doc_id, depth_limit=max_depth):
            data = self.G.edges[source, target]
            related.append({
                "source": source,
                "target": target,
                "relation": data.get("relation_type", ""),
                "source_name": self.G.nodes[source].get("name", source),
                "target_name": self.G.nodes[target].get("name", target),
            })

        # Also check reverse edges
        for source, target in nx.bfs_edges(self.G.reverse(), doc_id, depth_limit=max_depth):
            data = self.G.edges[target, source] if self.G.has_edge(target, source) else {}
            related.append({
                "source": target,
                "target": source,
                "relation": data.get("relation_type", ""),
                "source_name": self.G.nodes[target].get("name", target),
                "target_name": self.G.nodes[source].get("name", source),
            })

        return related

    def reasoning_path(self, source_id: str, target_id: str) -> list[dict]:
        """Find the shortest reasoning path between two legal entities."""
        try:
            path = nx.shortest_path(self.G, source_id, target_id)
            edges = []
            for i in range(len(path) - 1):
                data = self.G.edges[path[i], path[i + 1]]
                edges.append({
                    "from": path[i],
                    "to": path[i + 1],
                    "relation": data.get("relation_type", ""),
                    "from_name": self.G.nodes[path[i]].get("name", path[i]),
                    "to_name": self.G.nodes[path[i + 1]].get("name", path[i + 1]),
                })
            return edges
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def get_stats(self) -> dict:
        """Get graph statistics."""
        type_counts = {}
        for node, data in self.G.nodes(data=True):
            ntype = data.get("type", "UNKNOWN")
            type_counts[ntype] = type_counts.get(ntype, 0) + 1

        rel_counts = {}
        for source, target, data in self.G.edges(data=True):
            rtype = data.get("relation_type", "UNKNOWN")
            rel_counts[rtype] = rel_counts.get(rtype, 0) + 1

        return {
            "nodes": self.G.number_of_nodes(),
            "edges": self.G.number_of_edges(),
            "node_types": type_counts,
            "relation_types": rel_counts,
        }

    def save(self, path: Optional[str] = None):
        """Save graph to pickle file."""
        path = path or str(KG_PATH)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.G, f)
        print(f"Saved graph ({self.node_count} nodes, {self.edge_count} edges) -> {path}")

    def load(self, path: Optional[str] = None):
        """Load graph from pickle file."""
        path = path or str(KG_PATH)
        with open(path, "rb") as f:
            self.G = pickle.load(f)
        print(f"Loaded graph ({self.node_count} nodes, {self.edge_count} edges) <- {path}")