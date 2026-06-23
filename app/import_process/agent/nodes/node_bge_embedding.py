import sys
import os
from typing import Any, List, Dict

from app.import_process.agent.state import ImportGraphState
from app.lm.embedding_utils import get_bge_m3_ef, generate_embeddings
from app.utils.task_utils import add_running_task,add_done_task
from app.core.logger import logger

# BGE-M3 vectorization node.
def node_bge_embedding(state: ImportGraphState) -> ImportGraphState:
    """
    Generate dense and sparse BGE-M3 vectors for document chunks.
    """
    current_node = sys._getframe().f_code.co_name
    logger.info(f">>> Starting LangGraph node: {current_node}")

    add_running_task(state.get("task_id", ""), current_node)
    logger.info("--- BGE-M3 text vectorization started ---")

    try:
        texts_to_embed = step_1_validate_input(state)
        bge_m3_ef = step_2_init_model()
        output_data = step_3_generate_embeddings(texts_to_embed, bge_m3_ef)
        state['chunks'] = output_data
        logger.info(f"--- BGE-M3 vectorization finished. Processed {len(output_data)} chunks ---")
        add_done_task(state.get("task_id", ""), current_node)
    except Exception as e:
        logger.error(f"BGE-M3 vectorization node failed: {str(e)}", exc_info=True)

    return state

def step_1_validate_input(state: ImportGraphState) -> List[Dict[str, Any]]:
    """
    Validate the chunk list before vectorization.
    """
    texts_to_embed = state.get("chunks")
    if not isinstance(texts_to_embed, list) or not texts_to_embed:
        logger.error("Vectorization input validation failed: `chunks` is empty or not a valid list")
        raise ValueError("No valid text chunks are available for vectorization")

    logger.info(f"Vectorization input validation passed, chunk count: {len(texts_to_embed)}")
    return texts_to_embed

def step_2_init_model():
    """
    Initialize the singleton BGE-M3 model instance.
    """
    try:
        ef = get_bge_m3_ef()
        if ef is None:
            raise ValueError("BGE-M3 model instance is None: the model could not be loaded")

        logger.info("BGE-M3 model instance initialized successfully in singleton mode")
        return ef
    except Exception as e:
        error_msg = f"Failed to initialize the BGE-M3 model: {e}. Check the model path and environment configuration."
        logger.error(error_msg)
        raise ValueError(error_msg)

def step_3_generate_embeddings(texts_to_embed: List[Dict[str, Any]], bge_m3_ef: Any) -> List[Dict[str, Any]]:
    """
    Generate dense and sparse vectors in batches.
    """
    output_data = []
    batch_size = 5

    total = len(texts_to_embed)
    for i in range(0, total, batch_size):
        batch_texts = texts_to_embed[i:i + batch_size]
        start_idx, end_idx = i + 1, min(i + len(batch_texts), total)

        try:
            input_texts = []
            for doc in batch_texts:
                item_name = doc["item_name"]
                content = doc["content"]
                text = f"Item: {item_name}. Description: {content}" if item_name else content
                input_texts.append(text)


            docs_embeddings = generate_embeddings(input_texts)
            if not docs_embeddings:
                logger.warning(f"Chunks {start_idx}-{end_idx}: embedding generation returned empty output, keeping original data")
                output_data.extend(batch_texts)
                continue

            for j, doc in enumerate(batch_texts):
                item = doc.copy()
                item["dense_vector"] = docs_embeddings["dense"][j]
                item["sparse_vector"] = docs_embeddings["sparse"][j]
                output_data.append(item)

            logger.info(f"Chunks {start_idx}-{end_idx}: dense and sparse vectors generated successfully")

        except Exception as e:
            logger.error(
                f"Chunks {start_idx}-{end_idx}: vector generation failed, keeping original data | Error: {str(e)}",
                exc_info=True
            )
            output_data.extend(batch_texts)
            continue

    return output_data

# Local unit-test entry point.
if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    test_state = ImportGraphState({
        "task_id": "test_task_embedding_001",
        "chunks": [
            {
                "content": "This is test-document content used to verify that vectorization succeeds.",
                "title": "Test document title",
                "item_name": "Test item",
                "file_title": "test-file.pdf"
            },
            {
                "content": "This is the second test document, used to verify batch-processing logic.",
                "title": "Test document title 2",
                "item_name": "Test item",
                "file_title": "test-file.pdf"
            }
        ]
    })

    logger.info("=== Starting local unit test for the BGE-M3 vectorization node ===")
    try:
        result_state = node_bge_embedding(test_state)
        result_chunks = result_state.get("chunks", [])

        logger.info("=== Local vectorization-node test finished ===")
        logger.info(f"Test task ID: {test_state.get('task_id')}")
        logger.info(f"Expected chunk count: 2 | Actual processed chunk count: {len(result_chunks)}")
        logger.info(f"Vector payload: {result_chunks}")

        for idx, chunk in enumerate(result_chunks):
            has_dense = "dense_vector" in chunk
            has_sparse = "sparse_vector" in chunk
            logger.info(
                f"Chunk {idx + 1}: dense vector {'generated' if has_dense else 'missing'} | sparse vector {'generated' if has_sparse else 'missing'}")

    except Exception as e:
        logger.error(f"=== Local vectorization-node test failed === Error: {str(e)}", exc_info=True)
        logger.warning("Troubleshooting hint: check the BGE-M3 model path, available memory, and environment variables")
