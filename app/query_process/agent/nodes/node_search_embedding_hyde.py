# HyDE node.
import sys
from app.utils.task_utils import add_running_task, add_done_task
from app.lm.lm_utils import *
from app.lm.embedding_utils import *
from app.clients.milvus_utils import *
from app.core.logger import logger
from app.core.load_prompt import load_prompt
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


def step_1_create_hyde_doc(rewritten_query: str) -> str:
    """
    Stage 1: generate a hypothetical document from the rewritten query.
    """
    if not rewritten_query:
        logger.error("Step 1 error: rewritten_query is empty")
        raise ValueError("rewritten_query cannot be empty")

    logger.info(f"Step 1: Starting hypothetical-document generation (HyDE), query: {rewritten_query}")

    try:
        llm = get_llm_client()
        hyde_prompt = load_prompt("hyde_prompt", rewritten_query=rewritten_query)
        logger.debug(f"Step 1: Prompt loaded successfully, length: {len(hyde_prompt)}")

        response = llm.invoke(hyde_prompt)
        hyde_doc = response.content
        
        logger.info(f"Step 1: Hypothetical document generated, length: {len(hyde_doc)} characters")
        logger.debug(f"Step 1: Document preview: {hyde_doc[:50]}...")
        
        return hyde_doc

    except Exception as e:
        logger.error(f"Step 1: Failed to generate the hypothetical document: {e}")
        raise e


def step_2_search_embedding_hyde(
    rewritten_query: str,
    hyde_doc: str,
    item_names=None,
    req_limit: int = 10,
    top_k: int = 5,
    ranker_weights=(0.8, 0.2),  # Default weights biased toward dense vectors.
    norm_score: bool = True,    # Normalize scores by default.
    output_fields=["chunk_id", "content", "item_name"],
):
    """
    Stage 2: embed `rewritten_query + hyde_doc` and retrieve matching chunks from Milvus.
    """
    if not rewritten_query:
        raise ValueError("rewritten_query cannot be empty")
    if not hyde_doc:
        raise ValueError("hypothetical_doc cannot be empty")

    combined_text = rewritten_query + " " + hyde_doc
    logger.info(f"Step 2: Combined query + HyDE doc, total length: {len(combined_text)}")

    logger.info("Step 2: Generating hybrid embeddings...")
    embeddings = generate_embeddings([combined_text])
    
    collection_name = os.environ.get("CHUNKS_COLLECTION")
    if not collection_name:
        logger.error("Step 2 error: CHUNKS_COLLECTION is not set")
        return []
        
    logger.info(f"Step 2: Preparing hybrid retrieval in collection '{collection_name}'")

    # Build an optional filter expression.
    expr = None
    if item_names:
        quoted = ", ".join(f'"{v}"' for v in item_names)
        expr = f"item_name in [{quoted}]"
        logger.info(f"Step 2: Applying filter: {expr}")
    else:
        logger.info("Step 2: No item-name filter provided, searching the full collection")

    try:
        reqs = create_hybrid_search_requests(
            dense_vector=embeddings.get("dense")[0],
            sparse_vector=embeddings.get("sparse")[0],
            expr=expr,
            limit=req_limit,
        )

        client = get_milvus_client()
        if not client:
            logger.error("Step 2 error: unable to connect to Milvus")
            return []

        logger.info(f"Step 2: Running hybrid search, weights={ranker_weights}, top_k={top_k}")
        res = hybrid_search(
            client=client,
            collection_name=collection_name,
            reqs=reqs,
            ranker_weights=ranker_weights,
            norm_score=norm_score,
            limit=top_k,
            output_fields=list(output_fields),
        )
        
        hit_count = len(res[0]) if res and len(res) > 0 else 0
        logger.info(f"Step 2: Retrieval finished, found {hit_count} matching chunks")
        
        return res

    except Exception as e:
        logger.error(f"Step 2: Retrieval failed: {e}")
        return []


def node_search_embedding_hyde(state):
    """
    HyDE (Hypothetical Document Embedding) retrieval node.
    """
    logger.info("---HyDE node started---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    # 1. Extract and validate the query.
    rewritten_query = state.get("rewritten_query")
    if not rewritten_query:
        rewritten_query = state.get("original_query")
    
    if not rewritten_query:
        logger.error("HyDE node error: no valid user query was found")
        return {}

    item_names = state.get("item_names")
    logger.info(f"HyDE retrieval inputs: query='{rewritten_query}', item_names={item_names}")

    # Stage 1: generate the hypothetical document.
    hyde_doc = ""
    try:
        logger.info("Step 1: Starting HyDE document generation...")
        hyde_doc = step_1_create_hyde_doc(rewritten_query)
        logger.info(f"Step 1: HyDE document generated successfully (length: {len(hyde_doc)})")
        logger.debug(f"HyDE document preview: {hyde_doc[:100]}...")
    except Exception as e:
        logger.error(f"Step 1 (HyDE document generation) failed: {e}", exc_info=True)
        return {}

    # Stage 2: retrieve chunks using the rewritten question plus the HyDE doc.
    try:
        logger.info("Step 2: Running Milvus hybrid retrieval with the HyDE document...")
        res = step_2_search_embedding_hyde(
            rewritten_query=rewritten_query,
            hyde_doc=hyde_doc,
            item_names=item_names,
            top_k=5,
        )
        
        hit_count = len(res[0]) if res and len(res) > 0 else 0
        logger.info(f"Step 2: Retrieval finished, recalled {hit_count} relevant chunks")
        
        if hit_count > 0:
            first_hit = res[0][0]
            score = first_hit.get("distance")
            content_preview = first_hit.get("entity", {}).get("content", "")[:30]
            logger.debug(f"Top1 result: score={score}, content='{content_preview}...'")

        return {
            "hyde_embedding_chunks": res[0] if res else [],
            "hyde_doc": hyde_doc,
        }
    except Exception as e:
        logger.error(f"Step 2 (vector generation and retrieval) failed: {e}", exc_info=True)
        return {}
    finally:
        add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
        logger.info("---HyDE node finished---")


if __name__ == "__main__":
    print("\n" + "="*50)
    print(">>> Starting local test for node_search_embedding_hyde")
    print("="*50)
    
    # Mock input state
    mock_state = {
        "session_id": "test_hyde_session_001",
        "original_query": "How do I operate the HAK 180 hot stamping machine?",
        "rewritten_query": "What are the detailed operating steps for the HAK 180 hot stamping machine?",
        "item_names": ["HAK 180 hot stamping machine"],
        "is_stream": False
    }

    try:
        result = node_search_embedding_hyde(mock_state)
        
        print("\n" + "="*50)
        print(">>> Test result summary:")
        print(f"HyDE Doc Generated: {bool(result.get('hyde_doc'))}")
        if result.get("hyde_doc"):
            print(f"Doc Preview: {result.get('hyde_doc')[:50]}...")
            
        chunks = result.get("hyde_embedding_chunks", [])
        print(f"Chunks Found: {len(chunks)}, chunk payload: {chunks}")
        if chunks:
            print(f"Top Chunk Score: {chunks[0].get('distance')}")
        print("="*50)

    except Exception as e:
        logger.exception(f"Uncaught exception during local test: {e}")
