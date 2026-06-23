# Basic imports: system modules, paths, and typing helpers.
import os
import sys
from typing import List, Dict, Any, Tuple

# Milvus client and datatype enum used for collection setup.
from pymilvus import DataType
# LangChain message classes for standardized LLM conversation payloads.
from langchain_core.messages import SystemMessage, HumanMessage

# Project modules
from app.import_process.agent.state import ImportGraphState
from app.clients.milvus_utils import get_milvus_client
from app.lm.lm_utils import get_llm_client
from app.lm.embedding_utils import generate_embeddings
from app.utils.normalize_sparse_vector import normalize_sparse_vector
from app.utils.task_utils import add_running_task
from app.core.logger import logger
from app.core.load_prompt import load_prompt
from app.utils.escape_milvus_string_utils import escape_milvus_string

# --- Configuration ---
# Use only the first few chunks so the LLM context does not grow too large.
DEFAULT_ITEM_NAME_CHUNK_K = 5
# Truncate each chunk individually to protect the prompt budget.
SINGLE_CHUNK_CONTENT_MAX_LEN = 800
# Upper bound for the full prompt context.
CONTEXT_TOTAL_MAX_CHARS = 2500


def step_1_get_inputs(state: ImportGraphState) -> Tuple[str, List[Dict]]:
    """
    Step 1: read and validate the inputs for item-name recognition.

    This step:
        1. Extracts the file title and chunk list from workflow state.
        2. Applies multiple fallbacks for missing values.
        3. Performs light type validation before downstream processing.
    """
    file_title = state.get("file_title", "") or state.get("file_name", "")
    chunks = state.get("chunks") or []

    if not file_title:
        if chunks and isinstance(chunks[0], dict):
            file_title = chunks[0].get("file_title", "")
            logger.warning("state does not contain a valid file_title; fell back to the first chunk's file_title")

    if not file_title:
        logger.warning("state is missing both file_title and file_name; LLM recognition quality may be reduced")

    if not isinstance(chunks, list) or not chunks:
        logger.warning("state['chunks'] is empty or not a list; item-name recognition cannot proceed")
        return file_title, []

    logger.info(f"Step 1: input validation completed. Retrieved {len(chunks)} valid text chunks")
    return file_title, chunks


def step_2_build_context(
    chunks: List[Dict],
    k: int = DEFAULT_ITEM_NAME_CHUNK_K,
    max_chars: int = CONTEXT_TOTAL_MAX_CHARS,
) -> str:
    """
    Step 2: build a standardized context string for LLM item-name recognition.

    Rules:
        1. Only use the first k chunks.
        2. Apply both per-chunk and total-context truncation.
        3. Format the context with numbered sections for readability.
        4. Skip empty or invalid chunks.
    """
    if not chunks:
        return ""

    parts: List[str] = []
    total_chars = 0

    for idx, chunk in enumerate(chunks[:k]):
        if not isinstance(chunk, dict):
            logger.debug(f"Chunk {idx + 1} is not a dict and has been skipped")
            continue

        chunk_title = chunk.get("title", "").strip()
        chunk_content = chunk.get("content", "").strip()

        if not (chunk_title or chunk_content):
            logger.debug(f"Chunk {idx + 1} is blank and has been skipped")
            continue

        if len(chunk_content) > SINGLE_CHUNK_CONTENT_MAX_LEN:
            chunk_content = chunk_content[:SINGLE_CHUNK_CONTENT_MAX_LEN]
            logger.debug(f"Chunk {idx + 1} content was truncated to {SINGLE_CHUNK_CONTENT_MAX_LEN} characters")

        piece = f"[Chunk {idx + 1}]\nTitle: {chunk_title}\nContent: {chunk_content}"
        parts.append(piece)
        total_chars += len(piece)

        if total_chars > max_chars:
            logger.info(f"Context is about to exceed the limit ({max_chars}); remaining chunks will be skipped")
            break

    context = "\n\n".join(parts).strip()
    final_context = context[:max_chars]
    logger.info(f"Step 2: context built successfully. Final length: {len(final_context)} characters")
    return final_context


def step_3_call_llm(file_title: str, context: str) -> str:
    """
    Step 3: call the LLM to identify the primary item or model name.

    Fallback strategy:
        - If context is empty, return file_title directly.
        - If the model returns nothing or errors, return file_title.
    """
    logger.info("Starting Step 3: calling the LLM to recognize the item name")

    if not context:
        logger.warning("Context is empty; skipping the LLM call and falling back to the file title")
        return file_title

    try:
        human_prompt = load_prompt("item_name_recognition", file_title=file_title, context=context)
        system_prompt = load_prompt("product_recognition_system")
        logger.debug(
            f"LLM prompts built successfully. System prompt length: {len(system_prompt)}, "
            f"human prompt length: {len(human_prompt)}"
        )

        llm = get_llm_client(json_mode=False)
        if not llm:
            logger.error("Failed to get an LLM client; falling back to the file title")
            return file_title

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]
        resp = llm.invoke(messages)

        item_name = getattr(resp, "content", "").strip()
        item_name = item_name.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")

        if not item_name:
            logger.warning("The LLM returned empty content; falling back to the file title")
            return file_title

        logger.info(f"Step 3: item name recognized successfully: {item_name}")
        return item_name

    except Exception as e:
        logger.error(f"Step 3: LLM call failed: {str(e)}", exc_info=True)
        return file_title


def step_4_update_chunks(state: ImportGraphState, chunks: List[Dict], item_name: str):
    """
    Step 4: write the recognized item name back into state and every chunk.
    """
    state["item_name"] = item_name
    for chunk in chunks:
        chunk["item_name"] = item_name
    state["chunks"] = chunks
    logger.info(f"Step 4: item_name backfilled into {len(chunks)} chunks. Value: {item_name}")


def step_5_generate_vectors(item_name: str) -> Tuple[Any, Any]:
    """
    Step 5: generate dense and sparse BGE-M3 embeddings for the item name.
    """
    logger.info(f"Starting Step 5: generating BGE-M3 embeddings for item name [{item_name}]")

    if not item_name:
        logger.warning("item_name is empty; skipping embedding generation")
        return None, None

    try:
        vector_result = generate_embeddings([item_name])
        if vector_result and "dense" in vector_result and "sparse" in vector_result:
            dense_vector = vector_result["dense"][0]
            sparse_vector = vector_result["sparse"][0]
            logger.info("Step 5: dense and sparse vectors generated successfully")
        else:
            logger.warning("Step 5: embedding utility returned no usable result")
            dense_vector, sparse_vector = None, None
    except Exception as e:
        logger.error(f"Step 5: embedding generation failed: {str(e)}", exc_info=True)
        dense_vector, sparse_vector = None, None

    return dense_vector, sparse_vector


def step_6_save_to_milvus(state: ImportGraphState, file_title: str, item_name: str, dense_vector, sparse_vector):
    """
    Step 6: persist the item name, file title, and embeddings into Milvus.
    """
    milvus_uri = os.environ.get("MILVUS_URL")
    collection_name = os.environ.get("ITEM_NAME_COLLECTION")

    if not all([milvus_uri, collection_name]):
        logger.warning("Milvus configuration is incomplete (MILVUS_URL/ITEM_NAME_COLLECTION); skipping persistence")
        return

    logger.info(f"Starting Step 6: saving item name [{item_name}] into Milvus collection [{collection_name}]")

    try:
        client = get_milvus_client()
        if not client:
            logger.error("Could not get a Milvus client; skipping persistence")
            return

        if not client.has_collection(collection_name=collection_name):
            logger.info(f"Collection [{collection_name}] does not exist. Creating schema and indexes.")
            schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
            schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True, auto_id=True)
            schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

            index_params = client.prepare_index_params()
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_vector_index",
                index_type="HNSW",
                metric_type="COSINE",
                params={"M": 16, "efConstruction": 200},
            )
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_vector_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"inverted_index_algo": "DAAT_MAXSCORE", "normalize": True, "quantization": "none"},
            )

            client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
            logger.info(f"Collection [{collection_name}] created successfully with vector indexes")

        clean_item_name = (item_name or "").strip()
        if clean_item_name:
            client.load_collection(collection_name=collection_name)
            safe_item_name = escape_milvus_string(clean_item_name)
            filter_expr = f'item_name=="{safe_item_name}"'
            client.delete(collection_name=collection_name, filter=filter_expr)
            logger.info(f"Idempotent cleanup completed for item name [{clean_item_name}]")

        data = {
            "file_title": file_title,
            "item_name": item_name,
        }
        if dense_vector is not None:
            data["dense_vector"] = dense_vector
        if sparse_vector is not None:
            data["sparse_vector"] = normalize_sparse_vector(sparse_vector)

        client.insert(collection_name=collection_name, data=[data])
        client.load_collection(collection_name=collection_name)

        state["item_name"] = item_name
        logger.info(f"Step 6: item name [{item_name}] saved successfully to [{collection_name}] with fields: {list(data.keys())}")

    except Exception as e:
        logger.error(f"Step 6: failed to persist data to Milvus: {str(e)}", exc_info=True)


def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """
    Core node: recognize the primary product or item name from document chunks.

    Flow:
        1. Extract and validate inputs
        2. Build LLM context
        3. Recognize the item name with the LLM
        4. Backfill the result into state and chunks
        5. Generate dense and sparse embeddings
        6. Store the result in Milvus
    """
    node_name = sys._getframe().f_code.co_name
    logger.info(f">>> Starting core node: [Item Name Recognition] {node_name}")
    add_running_task(state.get("task_id", ""), node_name)

    try:
        file_title, chunks = step_1_get_inputs(state)
        if not chunks:
            logger.warning(f">>> Node warning: {node_name} (no valid chunk data), skipping recognition")
            return state

        context = step_2_build_context(chunks)
        item_name = step_3_call_llm(file_title, context)
        step_4_update_chunks(state, chunks, item_name)
        dense_vector, sparse_vector = step_5_generate_vectors(item_name)
        step_6_save_to_milvus(state, file_title, item_name, dense_vector, sparse_vector)

        logger.info(f">>> Core node finished: [Item Name Recognition] {node_name}, recognized item: {item_name}")

    except Exception as e:
        logger.error(f">>> Core node failed: [Item Name Recognition] {node_name}, error: {str(e)}", exc_info=True)
        state["item_name"] = "Unknown item"

    return state


def test_node_item_name_recognition():
    """
    Local test for the item-name recognition node.

    This lets us validate the node independently without running the full LangGraph workflow.
    """
    logger.info("=== Starting local test for node_item_name_recognition ===")
    try:
        mock_state = ImportGraphState({
            "task_id": "test_task_123456",
            "file_title": "Huawei Mate60 Pro Phone User Manual",
            "file_name": "HuaweiMate60ProManual.pdf",
            "chunks": [
                {
                    "title": "Product Overview",
                    "content": (
                        "The Huawei Mate60 Pro is a flagship smartphone released by Huawei in 2023. "
                        "It uses the Kirin 9000S chip, supports satellite calling, and features a 6.82-inch display "
                        "with a 2700 x 1224 resolution."
                    ),
                },
                {
                    "title": "Camera Features",
                    "content": (
                        "The Huawei Mate60 Pro includes a 50 MP main camera, a 12 MP ultra-wide camera, "
                        "and a 48 MP telephoto camera. It supports 5x optical zoom and 100x digital zoom."
                    ),
                },
                {
                    "title": "Battery Specifications",
                    "content": (
                        "The battery capacity is 5000 mAh. The device supports 88W wired fast charging, "
                        "50W wireless fast charging, and reverse wireless charging."
                    ),
                },
            ],
        })

        result_state = node_item_name_recognition(mock_state)

        logger.info("=== Local test for node_item_name_recognition completed ===")
        logger.info(f"Test task ID: {result_state.get('task_id')}")
        logger.info(f"Recognized item name: {result_state.get('item_name')}")
        logger.info(f"Chunk count: {len(result_state.get('chunks', []))}")
        logger.info(f"First chunk item_name: {result_state.get('chunks', [{}])[0].get('item_name')}")

        milvus_client = get_milvus_client()
        collection_name = os.environ.get("ITEM_NAME_COLLECTION")
        if milvus_client and collection_name:
            milvus_client.load_collection(collection_name)
            item_name = result_state.get("item_name")
            safe_name = escape_milvus_string(item_name)
            res = milvus_client.query(
                collection_name=collection_name,
                filter=f'item_name=="{safe_name}"',
                output_fields=["file_title", "item_name"],
            )
            logger.info(f"Queried data from Milvus: {res}")

    except Exception as e:
        logger.error(f"Local test for item-name recognition failed: {str(e)}", exc_info=True)


if __name__ == "__main__":
    test_node_item_name_recognition()
