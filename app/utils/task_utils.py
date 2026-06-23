from typing import Dict, List
from .sse_utils import push_to_session

# ---------------------------
# In-memory task tracking (single process)
# ---------------------------
# key: task_id
# value: list of node names (raw English names / node IDs)
_tasks_running_list: Dict[str, List[str]] = {}
_tasks_done_list: Dict[str, List[str]] = {}

# key: task_id
# value: status string, for example pending/processing/completed/failed
_tasks_status: Dict[str, str] = {}

# key: task_id
# value: task result, for example the answer from a query task
_tasks_result: Dict[str, Dict[str, str]] = {}

TASK_STATUS_PENDING = "pending"
TASK_STATUS_PROCESSING = "processing"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"

# Node-name to display-name mapping used by the frontend.
# Keys should match the node names used in LangGraph `add_node("xxx", ...)`.
_NODE_NAME_TO_CN: Dict[str, str] = {
    "upload_file": "Start file upload",
    "node_entry": "Check file",
    "node_pdf_to_md": "Convert PDF to Markdown",
    "node_md_img": "Process Markdown images",
    "node_item_name_recognition": "Recognize subject name",
    "node_document_split": "Split document",
    "node_bge_embedding": "Generate vectors",
    "node_import_kg": "Import knowledge graph",
    "node_import_milvus": "Import vector store",
    "__end__": "Completed",
    "END": "Completed",
    # --- Query flow nodes (kb/query_process/main_graph.py) ---
    "node_item_name_confirm": "Confirm product in question",
    "node_answer_output": "Generate answer",
    "node_rerank": "Rerank",
    "node_rrf": "Reciprocal-rank fusion",
    "node_web_search_mcp": "Web search",
    "node_search_embedding": "Chunk search",
    "node_search_embedding_hyde": "Chunk search (hypothetical document)",
    "node_multi_search": "Multi-route search",
    "node_query_kg": "Query knowledge graph",
    "node_join": "Merge multi-route search",
}


def _ensure_task(task_id: str) -> None:
    """Ensure the data structures for a task_id are initialized."""
    if task_id not in _tasks_running_list:
        _tasks_running_list[task_id] = []
    if task_id not in _tasks_done_list:
        _tasks_done_list[task_id] = []
    if task_id not in _tasks_result:
        _tasks_result[task_id] = {}


def _to_cn(node_name: str) -> str:
    """Convert a node name into its frontend display name."""
    return _NODE_NAME_TO_CN.get(node_name, node_name)


def add_running_task(task_id: str, node_name: str, is_stream: bool = False) -> None:
    """
    Add a running node task.

    Args:
    - task_id: Task ID
    - node_name: Node name (node ID)
    """
    _ensure_task(task_id)
    running = _tasks_running_list[task_id]
    # Avoid duplicate entries.
    if node_name not in running:
        running.append(node_name)

    if is_stream:
        task_push_queue(task_id)


def add_done_task(task_id: str, node_name: str, is_stream: bool = False) -> None:
    """
    Add a completed node task.

    Note: when a task is marked completed, the matching running task is removed.

    Args:
    - task_id: Task ID
    - node_name: Node name (node ID)
    """
    _ensure_task(task_id)

    # 1) Remove every matching node from the running list.
    running = _tasks_running_list[task_id]
    _tasks_running_list[task_id] = [n for n in running if n != node_name]

    # 2) Append to done while preserving completion order.
    done = _tasks_done_list[task_id]
    if node_name not in done:
        done.append(node_name)

    if is_stream:
        task_push_queue(task_id)


def set_task_result(task_id: str, key: str, value: str) -> None:
    """
    Store a task result field such as `answer` or `error`.
    """
    _ensure_task(task_id)
    _tasks_result[task_id][key] = value


def get_task_result(task_id: str, key: str, default: str = "") -> str:
    """
    Retrieve a task result field such as `answer` or `error`.
    """
    _ensure_task(task_id)
    return _tasks_result.get(task_id, {}).get(key, default)


def get_task_status(task_id: str) -> str:
    """
    Get the current task status.

    Args:
    - task_id: Task ID

    Returns:
    - str: Status name, or an empty string if it has not been set
    """
    return _tasks_status.get(task_id, "")


def get_done_task_list(task_id: str) -> List[str]:
    """
    Get the completed node list using display names.


    """
    _ensure_task(task_id)
    done = _tasks_done_list.get(task_id, [])
    return [_to_cn(n) for n in done]


def get_running_task_list(task_id: str) -> List[str]:
    """
    Get the running node list using display names.

    """
    _ensure_task(task_id)
    running = _tasks_running_list.get(task_id, [])
    return [_to_cn(n) for n in running]


def update_task_status(task_id: str, status_name: str, push_queue: bool = False) -> None:
    """
    Update the task status.

    Args:
    - task_id: Task ID
    - status_name: Status name string
    """
    _tasks_status[task_id] = status_name
    if push_queue:
        task_push_queue(task_id)


def task_push_queue(task_id: str):
    push_to_session(task_id, "progress", {
        "status": get_task_status(task_id),
        "done_list": get_done_task_list(task_id),
        "running_list": get_running_task_list(task_id),
    })


#
def clear_task(task_id: str):
    _tasks_running_list.pop(task_id, None)
    _tasks_done_list.pop(task_id, None)
    _tasks_status.pop(task_id, None)
    _tasks_result.pop(task_id, None)
