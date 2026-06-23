from langgraph.graph import StateGraph, END
from app.query_process.agent.state import QueryGraphState
# Import all node functions.
from app.query_process.agent.nodes.node_item_name_confirm import node_item_name_confirm
from app.query_process.agent.nodes.node_query_kg import node_query_kg
from app.query_process.agent.nodes.node_answer_output import node_answer_output
from app.query_process.agent.nodes.node_rerank import node_rerank
from app.query_process.agent.nodes.node_rrf import node_rrf
from app.query_process.agent.nodes.node_search_embedding import node_search_embedding
from app.query_process.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.query_process.agent.nodes.node_web_search_mcp import node_web_search_mcp

# Initialize the state graph.
builder = StateGraph(QueryGraphState)

# Register all nodes.
builder.add_node("node_item_name_confirm", node_item_name_confirm)  # Confirm the product in question
builder.add_node("node_multi_search", lambda x: x)  # Virtual node: multi-route search split point
builder.add_node("node_search_embedding", node_search_embedding)  # Embedding search
builder.add_node("node_search_embedding_hyde", node_search_embedding_hyde)
builder.add_node("node_query_kg", node_query_kg)
builder.add_node("node_web_search_mcp", node_web_search_mcp)
builder.add_node("node_join", lambda x: {})  # Virtual node: multi-route merge point
builder.add_node("node_rrf", node_rrf)  # Ranking
builder.add_node("node_rerank", node_rerank)  # Reranking
builder.add_node("node_answer_output", node_answer_output)  # Answer generation

# Virtual nodes serve as split / merge transfer points in the workflow.
# `lambda x: x` is the lightest possible pass-through implementation.
# A named function such as `def fn(state): return state` would be equivalent.

# Set the entry point.
builder.set_entry_point("node_item_name_confirm")


def route_after_item_confirm(state: QueryGraphState):
    # If an answer already exists, skip retrieval and jump straight to output.
    if state.get("answer"):
        return "node_answer_output"
    # Otherwise continue into the retrieval pipeline.
    return "node_multi_search"


# 1. Intent confirmation -> conditional branch -> retrieval or answer output
builder.add_conditional_edges(
    "node_item_name_confirm",
    route_after_item_confirm
)

# 2. Run four retrieval routes in parallel
builder.add_edge("node_multi_search", "node_search_embedding")
builder.add_edge("node_multi_search", "node_search_embedding_hyde")
builder.add_edge("node_multi_search", "node_web_search_mcp")
builder.add_edge("node_multi_search", "node_query_kg")

# 3. Merge the four retrieval branches
builder.add_edge("node_search_embedding", "node_join")
builder.add_edge("node_search_embedding_hyde", "node_join")
builder.add_edge("node_web_search_mcp", "node_join")
builder.add_edge("node_query_kg", "node_join")

# 4. Merge -> rank -> rerank -> generate -> end
builder.add_edge("node_join", "node_rrf")
builder.add_edge("node_rrf", "node_rerank")
builder.add_edge("node_rerank", "node_answer_output")
builder.add_edge("node_answer_output", END)

# Compile the runnable application.
query_app = builder.compile()
