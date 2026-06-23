from typing import TypedDict
import copy
from app.core.logger import logger

class ImportGraphState(TypedDict):
    """
    State definition for the import graph.
    Includes every field produced or consumed by the workflow nodes.
    """
    task_id: str          # Unique task ID used for tracing logs

    # --- Flow-control flags ---
    is_md_read_enabled: bool   # Whether the direct Markdown path is enabled
    is_pdf_read_enabled: bool  # Whether the PDF import path is enabled


    # --- Chunking options ---
    is_normal_split_enabled: bool
    is_silicon_flow_api_enabled: bool
    is_advanced_split_enabled: bool
    is_vllm_enabled: bool

    # --- Path fields ---
    local_dir: str        # Current working or output directory
    local_file_path: str  # Original input file path
    file_title: str       # File title, usually the stem without extension
    pdf_path: str         # PDF file path when the input is a PDF
    md_path: str          # Markdown file path, converted or directly provided
    split_path: str       # Path to the generated chunk file
    embeddings_path: str  # Path related to vector data output

    # --- Content fields ---
    md_content: str       # Full Markdown content
    chunks: list          # Split text chunks with metadata
    item_name: str        # Recognized main item name used for downstream retrieval

    # --- Database fields ---
    embeddings_content: list # Vector records prepared for Milvus


# Default initial graph state.
graph_default_state: ImportGraphState = {
    "task_id":"",
    "is_pdf_read_enabled": False,
    "is_md_read_enabled": False,
    "is_normal_split_enabled": True,
    "is_silicon_flow_api_enabled": True,
    "is_advanced_split_enabled": False,
    "is_vllm_enabled": False,
    "local_dir": "",
    "local_file_path": "",
    "pdf_path": "",
    "md_path": "",
    "file_title": "",
    "split_path": "",
    "embeddings_path": "",
    "md_content": "",
    "chunks": [],
    "item_name": "",
    "embeddings_content": []
}

def create_default_state(**overrides) -> ImportGraphState:
    """
    Create a default state object with optional overrides.

    Args:
        **overrides: Fields to override

    Returns:
        New state instance

    Examples:
        state = create_default_state(task_id="task_001", local_file_path="doc.pdf")
    """

    # Start from the default state.
    state = copy.deepcopy(graph_default_state)
    # Apply any requested overrides.
    state.update(overrides)
    # Return the new state dictionary.
    return state

def get_default_state() -> ImportGraphState:
    """
    Return a fresh state instance and avoid mutating the global default.
    """
    return copy.deepcopy(graph_default_state)


if __name__ == "__main__":
    # Local test
    state = create_default_state(local_file_path="using-the-rs-12-multimeter.pdf")
    logger.info(state)
