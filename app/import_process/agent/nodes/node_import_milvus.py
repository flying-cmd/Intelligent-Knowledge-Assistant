import os
import sys
from typing import List, Dict, Any

# Milvus dependencies
from pymilvus import DataType

# Project modules
from app.import_process.agent.state import ImportGraphState
from app.clients.milvus_utils import get_milvus_client
from app.utils.task_utils import add_running_task
from app.core.logger import logger
from app.conf.milvus_config import milvus_config
from app.utils.escape_milvus_string_utils import escape_milvus_string

# Read the chunk collection name from config to keep environment setup decoupled.
CHUNKS_COLLECTION_NAME = milvus_config.chunks_collection

# ==========================================
# Core Milvus chunk-ingestion node
# Responsibilities:
#   1. Idempotency: delete old data for the same item_name before insert.
#   2. Auto-create the collection, schema, and indexes when missing.
#   3. Validate chunk/vector fields before insertion.
#   4. Backfill Milvus-generated chunk_id values for downstream use.
# Upstream dependency:
#   BGE-M3 embedding node providing dense_vector and sparse_vector fields.
# ==========================================
def node_import_milvus(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: insert chunk data into Milvus.

    Flow:
        1. Validate inputs and extract vector dimensions.
        2. Connect to Milvus and prepare the collection.
        3. Remove old data for the same item_name.
        4. Insert the new chunks in batch and backfill chunk_id.
        5. Update the global state with the inserted chunks.
    """
    current_node = sys._getframe().f_code.co_name
    logger.info(f">>> Starting LangGraph node: {current_node} (Milvus chunk ingestion)")
    add_running_task(state["task_id"], current_node)
    logger.info("--- Milvus chunk ingestion started ---")

    try:
        chunks_json_data, vector_dimension = step_1_check_input(state)
        client = step_2_prepare_collection(vector_dimension)
        step_3_clean_old_data(client, chunks_json_data)
        updated_chunks = step_4_insert_data(client, chunks_json_data)
        state["chunks"] = updated_chunks
        logger.info("--- Milvus chunk ingestion completed ---")
    except Exception as e:
        logger.error(f"Milvus chunk ingestion node failed: {str(e)}", exc_info=True)
        raise ValueError(f"An error occurred during Milvus import: {e}")

    return state


def step_1_check_input(state: Dict[str, Any]) -> tuple[List[Dict[str, Any]], int]:
    """
    Step 1: validate the chunk data before insertion.

    Checks:
        1. chunks must be present and be a non-empty list.
        2. The first chunk must contain dense_vector.
        3. Extract the dense-vector dimension for collection setup.
    """
    chunks_json_data = state.get("chunks")
    if not chunks_json_data:
        logger.error("Milvus validation failed: state['chunks'] is empty")
        raise ValueError("Error: chunks is empty, so Milvus ingestion cannot proceed")
    if not isinstance(chunks_json_data, list) or len(chunks_json_data) == 0:
        logger.error("Milvus validation failed: chunks is not a non-empty list")
        raise ValueError("Error: chunks must be a non-empty list")

    first_chunk = chunks_json_data[0]
    if "dense_vector" not in first_chunk:
        logger.error("Milvus validation failed: chunk is missing dense_vector, upstream embedding may have failed")
        raise ValueError("Error: dense_vector is missing. Check the upstream embedding node")

    vector_dimension = len(first_chunk["dense_vector"])
    item_name = first_chunk.get("item_name", "Unknown item")
    logger.info(
        f"Milvus validation passed. Chunk count: {len(chunks_json_data)} | "
        f"Vector dimension: {vector_dimension} | Item name: {item_name}"
    )
    return chunks_json_data, vector_dimension


def create_collection(client, collection_name: str, vector_dimension: int):
    """
    Helper: create the Milvus collection and indexes automatically.
    """
    schema = client.create_schema(auto_id=True, enable_dynamic_fields=True)

    schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)  # Chunk content
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=65535)  # Chunk title
    schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535)  # Parent title
    schema.add_field(field_name="part", datatype=DataType.INT8)  # Chunk index within parent section
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)  # Source file title
    schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)  # Idempotency key
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)  # Sparse vector
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=vector_dimension)  # Dense vector
    # For the BGE-M3 model, the output dimension is fixed at 1024.
    # If you switch models, the dimension must match that model's architecture.

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",
        index_name="dense_vector_index",
        index_type="HNSW",
        metric_type="COSINE",
        # M controls max graph connections; efConstruction controls build-time search breadth.
        params={"M": 16, "efConstruction": 200}
    )

    index_params.add_index(
        field_name="sparse_vector",
        index_name="sparse_vector_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
        # DAAT_MAXSCORE is an efficient sparse-retrieval algorithm.
        params={"inverted_index_algo": "DAAT_MAXSCORE", "normalize": True, "quantization": "none"}
    )

    client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
    logger.info(f"Milvus collection created successfully: {collection_name}, vector dimension: {vector_dimension}")


def step_2_prepare_collection(vector_dimension: int):
    """
    Step 2: connect to Milvus and prepare the target collection.
    """
    logger.info(f"Preparing Milvus environment for collection: {CHUNKS_COLLECTION_NAME}")
    client = get_milvus_client()
    if client is None:
        logger.error("Failed to get Milvus client: get_milvus_client() returned None")
        raise ValueError("Milvus connection failed: get_milvus_client() returned None")

    if not CHUNKS_COLLECTION_NAME:
        logger.error("Milvus collection name is not configured: CHUNKS_COLLECTION_NAME is empty")
        raise ValueError("CHUNKS_COLLECTION is not configured")

    if not client.has_collection(collection_name=CHUNKS_COLLECTION_NAME):
        logger.info(f"Collection {CHUNKS_COLLECTION_NAME} does not exist. Creating schema and indexes automatically.")
        create_collection(client, CHUNKS_COLLECTION_NAME, vector_dimension)
    else:
        logger.info(f"Collection {CHUNKS_COLLECTION_NAME} already exists and will be reused")

    return client


def step_3_clean_old_data(client, chunks_json_data: List[Dict[str, Any]]):
    """
    Step 3: perform idempotent cleanup by deleting old data for the same item_name values.
    """
    # The walrus operator assigns and filters item_name values in a single expression.
    item_names = sorted(
        {
            name
            for x in chunks_json_data or []
            if (name := str(x.get("item_name", "")).strip())
        }
    )

    if not item_names:
        logger.warning("Skipping Milvus idempotent cleanup: no valid item_name was found in the chunks")
        return
    if len(item_names) > 1:
        logger.warning(f"Multiple item_name values detected for cleanup. Processing each one: {item_names}")

    for i_name in item_names:
        _clear_chunks_by_item_name(client, CHUNKS_COLLECTION_NAME, i_name)


def _clear_chunks_by_item_name(client, collection_name: str, item_name: str):
    """
    Internal helper: delete old chunk data from Milvus for a given item_name.
    """
    i_name = (item_name or "").strip()
    if not i_name:
        logger.warning("Skipping single-item cleanup: item_name is empty")
        return
    if not collection_name:
        logger.warning("Skipping single-item cleanup: collection name is not configured")
        return

    try:
        if not client.has_collection(collection_name=collection_name):
            logger.info(f"Skipping single-item cleanup: collection {collection_name} does not exist")
            return

        safe_item_name = escape_milvus_string(i_name)
        filter_expr = f'item_name == "{safe_item_name}"'
        logger.info(f"Starting idempotent cleanup in {collection_name} for item_name={i_name}")

        client.delete(collection_name=collection_name, filter=filter_expr)

        if hasattr(client, "flush"):
            try:
                client.flush(collection_name=collection_name)
            except Exception as e:
                logger.warning(f"Flush failed during cleanup, but the main flow can continue. Error: {str(e)}")

        logger.info(f"Idempotent cleanup finished successfully for item_name={i_name}")
    except Exception as e:
        logger.error(f"Idempotent cleanup failed for item_name={i_name} | Error: {str(e)}", exc_info=True)
        raise ValueError(f"Idempotent cleanup failed (item_name={i_name}): {e}")


def step_4_insert_data(client, chunks_json_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Step 4: insert chunk data into Milvus in batch and backfill chunk_id values.
    """
    data_to_insert = []
    for item in chunks_json_data:
        item_copy = item.copy()
        if isinstance(item_copy, dict) and "chunk_id" in item_copy:
            item_copy.pop("chunk_id", None)
        data_to_insert.append(item_copy)

    logger.info(f"Milvus insert started: preparing to insert {len(data_to_insert)} chunk records")
    insert_result = client.insert(collection_name=CHUNKS_COLLECTION_NAME, data=data_to_insert)
    insert_count = insert_result.get("insert_count", 0)
    logger.info(f"Milvus insert completed: inserted {insert_count} records, result={insert_result}")

    inserted_ids = insert_result.get("ids", [])
    if inserted_ids and len(inserted_ids) == len(chunks_json_data):
        logger.info(f"Backfilling {len(inserted_ids)} auto-generated chunk_id values")
        for idx, item in enumerate(chunks_json_data):
            item["chunk_id"] = str(inserted_ids[idx])
        logger.info("chunk_id backfill completed for all chunks")
    else:
        logger.warning(
            f"chunk_id backfill failed: generated ID count ({len(inserted_ids)}) "
            f"does not match chunk count ({len(chunks_json_data)})"
        )

    return chunks_json_data


if __name__ == "__main__":
    # Local test for the full Milvus import flow: connect, create collection, clean old data, and insert new data.
    import sys
    import os
    from dotenv import load_dotenv

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    load_dotenv(os.path.join(project_root, ".env"))

    dim = 1024
    test_state = {
        "task_id": "test_milvus_task",
        "chunks": [
            {
                "content": "Milvus test text 1",
                "title": "Test title",
                "item_name": "Test_Item_Milvus",  # Required for idempotent cleanup
                "parent_title": "test.pdf",
                "part": 1,
                "file_title": "test.pdf",
                "dense_vector": [0.1] * dim,  # Mock dense vector
                "sparse_vector": {1: 0.5, 10: 0.8}  # Mock sparse vector
            }
        ]
    }

    print("Running the Milvus import node test...")
    try:
        if not os.getenv("MILVUS_URL"):
            print("MILVUS_URL is not set, so Milvus cannot be reached")
        elif not os.getenv("CHUNKS_COLLECTION"):
            print("CHUNKS_COLLECTION is not set")
        else:
            result_state = node_import_milvus(test_state)
            chunks = result_state.get("chunks", [])
            if chunks and chunks[0].get("chunk_id"):
                print(f"Milvus import test passed, generated ID: {chunks[0]['chunk_id']}")
            else:
                print("Test failed: chunk_id was not returned")

    except Exception as e:
        print(f"Test failed: {e}")
