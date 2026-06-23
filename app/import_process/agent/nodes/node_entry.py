import os
import sys
from os.path import splitext

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.format_utils import format_state
from app.utils.task_utils import add_running_task, add_done_task

def node_entry(state: ImportGraphState) -> ImportGraphState:
    """
    Entry node for the LangGraph knowledge-base import workflow.
    Responsibilities: validate input, detect file type, enable the proper path,
    and extract the file title used downstream.
    """

    # Resolve the function name dynamically for logging.
    func_name = sys._getframe().f_code.co_name

    logger.debug(f"[{func_name}] Node started.\nCurrent workflow state: {format_state(state)}")

    # Mark the node as running.
    add_running_task(state["task_id"], func_name)


    # 1. Validate the input file path.
    document_path = state.get("local_file_path", "")
    if not document_path:
        logger.error(f"[{func_name}] Missing required parameter: `local_file_path` is empty")
        return state

    # 2. Enable the appropriate import path based on the file extension.
    if document_path.endswith(".pdf"):
        logger.info(f"[{func_name}] File type detected: {document_path} -> PDF, enabling the PDF workflow")
        state["is_pdf_read_enabled"] = True
        state["pdf_path"] = document_path
    elif document_path.endswith(".md"):
        logger.info(f"[{func_name}] File type detected: {document_path} -> Markdown, enabling the Markdown workflow")
        state["is_md_read_enabled"] = True
        state["md_path"] = document_path
    else:
        logger.warning(f"[{func_name}] Unsupported file type: {document_path}. Only .pdf and .md are supported")

    # 3. Extract the filename stem used as a business identifier.
    file_name = os.path.basename(document_path)
    state["file_title"] = splitext(file_name)[0]
    logger.info(f"[{func_name}] Extracted file_title: {state['file_title']}")

    # Mark the node as completed.
    add_done_task(state["task_id"], func_name)

    logger.debug(f"[{func_name}] Node finished.\nUpdated workflow state: {format_state(state)}")

    return state

if __name__ == '__main__':

    # Local unit tests covering unsupported, Markdown, and PDF inputs.
    logger.info("===== Starting node_entry unit test =====")

    # Test 1: unsupported TXT file
    test_state1 = create_default_state(
        task_id="test_task_001",
        local_file_path="lenovo-dolphin-user-manual.txt"
    )
    node_entry(test_state1)

    # Test 2: Markdown file
    test_state2 = create_default_state(
        task_id="test_task_002",
        local_file_path="xiaomi-user-manual.md"
    )
    node_entry(test_state2)

    # Test 3: PDF file
    test_state3 = create_default_state(
        task_id="test_task_003",
        local_file_path="using-the-multimeter.pdf"
    )
    node_entry(test_state3)

    logger.info("===== Finished node_entry unit test =====")
