"""Lightweight local knowledge graph (subject -> relation -> object) using NetworkX."""
import networkx as nx

# Seed triples so the graph has something to query out of the box.
SEED_TRIPLES = [
    ("RAG", "uses", "retriever"),
    ("RAG", "uses", "language_model"),
    ("FAISS", "is_a", "vector_database"),
    ("FAISS", "used_for", "similarity_search"),
    ("LangGraph", "is_a", "orchestration_framework"),
    ("LangGraph", "built_with", "nodes_and_edges"),
    ("NetworkX", "is_a", "graph_library"),
    ("NetworkX", "alternative_to", "Neo4j"),
    ("knowledge_graph", "stores", "entities_and_relationships"),
    ("sentence-transformers", "produces", "embeddings"),
]


def build_graph(triples=None) -> nx.DiGraph:
    triples = triples or SEED_TRIPLES
    g = nx.DiGraph()
    for subj, rel, obj in triples:
        g.add_edge(subj, obj, relation=rel)
    return g


def add_triple(g: nx.DiGraph, subj: str, rel: str, obj: str):
    g.add_edge(subj, obj, relation=rel)


def query_entity(g: nx.DiGraph, entity: str) -> list[str]:
    """Return facts about an entity as readable strings, if it exists in the graph."""
    if entity not in g:
        return []
    facts = []
    for _, obj, data in g.out_edges(entity, data=True):
        facts.append(f"{entity} {data['relation']} {obj}")
    for subj, _, data in g.in_edges(entity, data=True):
        facts.append(f"{subj} {data['relation']} {entity}")
    return facts


def find_mentioned_entity(g: nx.DiGraph, text: str) -> str | None:
    """Naive entity match: check if any graph node appears in the query text."""
    lowered = text.lower()
    for node in g.nodes:
        if node.lower().replace("_", " ") in lowered:
            return node
    return None
