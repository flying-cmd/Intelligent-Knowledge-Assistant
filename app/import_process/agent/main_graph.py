# Load environment variables before reading runtime configuration.
from dotenv import load_dotenv
# LangGraph core types: StateGraph plus built-in START/END markers.
from langgraph.graph import StateGraph, END, START

from app.core.logger import logger
# Shared workflow state used across all import nodes.
from app.import_process.agent.state import ImportGraphState, create_default_state
# Workflow nodes for the knowledge-base import pipeline.
from app.import_process.agent.nodes.node_entry import node_entry  # Entry node: initialize parameters and validate input
from app.import_process.agent.nodes.node_pdf_to_md import node_pdf_to_md  # Convert PDF files to Markdown
from app.import_process.agent.nodes.node_md_img import node_md_img  # Process Markdown images and fix image paths
from app.import_process.agent.nodes.node_document_split import node_document_split  # Split long documents into chunks
from app.import_process.agent.nodes.node_item_name_recognition import node_item_name_recognition  # Extract the main item name
from app.import_process.agent.nodes.node_bge_embedding import node_bge_embedding  # Convert text chunks into embeddings
from app.import_process.agent.nodes.node_import_milvus import node_import_milvus  # Write embeddings into Milvus


# Initialize environment variables up front.
load_dotenv()

# ===================== 1. Build the LangGraph state graph =====================
workflow = StateGraph(ImportGraphState)

# ===================== 2. Register workflow nodes =====================
workflow.add_node("node_entry", node_entry)
workflow.add_node("node_pdf_to_md", node_pdf_to_md)
workflow.add_node("node_md_img", node_md_img)
workflow.add_node("node_document_split", node_document_split)
workflow.add_node("node_item_name_recognition", node_item_name_recognition)
workflow.add_node("node_bge_embedding", node_bge_embedding)
workflow.add_node("node_import_milvus", node_import_milvus)

# ===================== 3. Set the workflow entry point =====================
workflow.set_entry_point("node_entry")

# ===================== 4. Define routing after the entry node =====================
def route_after_entry(state: ImportGraphState) -> str:
    """
    Route to the next node based on the enabled import mode.
    :param state: Full workflow state
    :return: Target node name or `END`
    """
    # Branch 1: direct Markdown import.
    if state.get("is_md_read_enabled"):
        return "node_md_img"
    # Branch 2: PDF import path.
    elif state.get("is_pdf_read_enabled"):
        return "node_pdf_to_md"
    # Branch 3: no import path enabled.
    else:
        return END

# Register the conditional routing edges from the entry node.
workflow.add_conditional_edges(
    "node_entry",
    route_after_entry,
    {
        "node_md_img": "node_md_img",
        "node_pdf_to_md": "node_pdf_to_md",
        END: END
    }
)

# ===================== 5. Register the shared sequential edges =====================
workflow.add_edge("node_pdf_to_md", "node_md_img")
workflow.add_edge("node_md_img", "node_document_split")
workflow.add_edge("node_document_split", "node_item_name_recognition")
workflow.add_edge("node_item_name_recognition", "node_bge_embedding")
workflow.add_edge("node_bge_embedding", "node_import_milvus")
workflow.add_edge("node_import_milvus", END)

# ===================== 6. Compile the graph into an executable app =====================
kb_import_app = workflow.compile()
