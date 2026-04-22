"""Knowledge Graph visualization using Pyvis + NetworkX."""

from pathlib import Path
from typing import Optional

from src.knowledge.graph_builder import LegalKnowledgeGraph


def visualize_graph(
    kg: LegalKnowledgeGraph,
    output_path: str = "legal_kg.html",
    max_nodes: int = 100,
    focus_node: Optional[str] = None,
    height: str = "800px",
):
    """Create interactive HTML visualization of the legal knowledge graph.

    Args:
        kg: LegalKnowledgeGraph instance
        output_path: Path to output HTML file
        max_nodes: Max nodes to display (subgraph if larger)
        focus_node: Center visualization on this node
        height: Height of the visualization
    """
    try:
        from pyvis.network import Network
    except ImportError:
        raise ImportError("pyvis not installed. Run: pip install pyvis")

    import networkx as nx

    # Select subgraph if too large
    if kg.node_count > max_nodes:
        if focus_node and focus_node in kg.G:
            # Ego graph around focus node
            subG = nx.ego_graph(kg.G, focus_node, radius=2)
        else:
            # Take most connected nodes
            degree_dict = dict(kg.G.degree())
            top_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)[:max_nodes]
            subG = kg.G.subgraph(top_nodes)
    else:
        subG = kg.G

    # Create Pyvis network
    net = Network(
        height=height,
        width="100%",
        directed=True,
        notebook=False,
        heading="Legal Knowledge Graph — Vietnamese Legal Documents",
    )

    # Add nodes with type-based styling
    for node, data in subG.nodes(data=True):
        ntype = data.get("type", "UNKNOWN")
        color = kg.NODE_COLORS.get(ntype, "#999999")
        label = data.get("name", node)
        # Truncate long labels
        if len(label) > 40:
            label = label[:37] + "..."

        title = f"Type: {ntype}\nName: {data.get('name', node)}"
        if data.get("effective_date"):
            title += f"\nEffective: {data['effective_date']}"
        if data.get("issuer"):
            title += f"\nIssuer: {data['issuer']}"

        net.add_node(
            node,
            label=label,
            title=title,
            color=color,
            size=max(10, min(50, subG.degree(node) * 5 + 10)),
            group=ntype,
        )

    # Add edges with relation-based styling
    for source, target, data in subG.edges(data=True):
        rtype = data.get("relation_type", "")
        color = kg.EDGE_COLORS.get(rtype, "#AAAAAA")
        label = rtype.replace("_", " ") if rtype else ""
        # Dashed line for destructive relations
        dashes = rtype in ("HET_HIEU_LUC", "THAY_THE")

        net.add_edge(
            source, target,
            label=label,
            color=color,
            dashes=dashes,
            title=f"{source} → {target}\nRelation: {rtype}",
        )

    # Physics layout
    net.force_atlas_2based()
    net.show_buttons(filter_=["physics"])

    # Save
    output_path = str(Path(output_path))
    net.save_graph(output_path)
    print(f"Visualization saved → {output_path}")
    return output_path


def visualize_amendment_chain(kg: LegalKnowledgeGraph, doc_id: str, output_path: str = "amendment_chain.html"):
    """Visualize the amendment chain for a specific document."""
    try:
        from pyvis.network import Network
    except ImportError:
        raise ImportError("pyvis not installed. Run: pip install pyvis")

    # Get amendment chain
    chain = kg.amendment_chain(doc_id)

    # Build subgraph
    nodes = {doc_id}
    for item in chain:
        nodes.add(item["amending_doc"])
        nodes.add(item["amended_doc"])

    if doc_id in kg.G:
        nodes.add(doc_id)

    subG = kg.G.subgraph([n for n in nodes if n in kg.G])

    net = Network(
        height="600px", width="100%", directed=True,
        heading=f"Amendment Chain: {kg.G.nodes[doc_id].get('name', doc_id)}",
    )

    for node, data in subG.nodes(data=True):
        ntype = data.get("type", "UNKNOWN")
        color = kg.NODE_COLORS.get(ntype, "#999999")
        net.add_node(node, label=data.get("name", node)[:30], color=color, group=ntype)

    for source, target, data in subG.edges(data=True):
        rtype = data.get("relation_type", "")
        color = kg.EDGE_COLORS.get(rtype, "#AAAAAA")
        net.add_edge(source, target, label=rtype.replace("_", " "), color=color)

    net.force_atlas_2based()
    net.save_graph(output_path)
    print(f"Amendment chain visualization saved → {output_path}")