import sys
import os
from app.utils.task_utils import add_running_task,add_done_task
from app.lm.embedding_utils import generate_embeddings
from app.clients.milvus_utils import create_hybrid_search_requests,hybrid_search,get_milvus_client
from app.core.logger import logger
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())


def node_search_embedding(state):
    """
    Run Milvus hybrid retrieval using the confirmed product names plus the
    rewritten user query.
    """
    logger.info("---search_milvus started---")
    add_running_task(state["session_id"],sys._getframe().f_code.co_name,state["is_stream"])

    # 1. Extract the key inputs used for retrieval.
    query = state.get("rewritten_query")
    item_names = state.get("item_names")
    
    logger.info(f"Core inputs extracted: query='{query}', item_names={item_names}")

    # 2. Vectorize the rewritten question into BGEM3 dense and sparse vectors.
    logger.info(f"Generating embeddings for text: {query[:50]}..." if len(query) > 50 else f"Generating embeddings for: '{query}'")
    embeddings = generate_embeddings([query])
    
    dense_vec = embeddings.get("dense")[0]
    sparse_vec = embeddings.get("sparse")[0]
    logger.debug(f"Embedding generation succeeded: dense_dim={len(dense_vec)}, sparse_len={len(sparse_vec)}")

    # 3. Resolve the target Milvus collection.
    collection_name = os.environ.get("CHUNKS_COLLECTION")
    logger.info(f"Connecting to Milvus and preparing collection '{collection_name}'...")

    # 4. Build the filtered hybrid-search requests.
    if not item_names:
        logger.warning("item_names is empty, skipping retrieval and returning no results")
        return {"embedding_chunks": []}
        
    quoted = ", ".join(f'"{v}"' for v in item_names)
    expr = f"item_name in [{quoted}]"
    logger.info(f"Built search filter expression: {expr}")

    reqs = create_hybrid_search_requests(
        dense_vector=dense_vec,
        sparse_vector=sparse_vec,
        expr=expr,
        limit=10
    )

    # 5. Run the Milvus hybrid retrieval.
    logger.info("Running Milvus hybrid retrieval...")
    client = get_milvus_client()
    res = hybrid_search(
        client=client,
        collection_name=collection_name,
        reqs=reqs,
        ranker_weights=(0.8, 0.2),
        norm_score=True,
        limit=5,
        output_fields=["chunk_id", "content", "item_name"]
    )

    hit_count = len(res[0]) if res and len(res) > 0 else 0
    logger.info(f"node_search_embedding completed successfully and retrieved {hit_count} matching chunks")
    if hit_count > 0:
        logger.debug(f"Top 1 retrieval result example: {res[0][0]}")
        
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    # 6. Return the first result list for this single-query search.
    return {"embedding_chunks": res[0] if res else []}


if __name__ == "__main__":
    # Mock test data
    test_state = {
        "session_id": "test_search_embedding_001",
        "rewritten_query": "HAK 180 hot stamping machine user guide",
        "item_names": ["HAK 180 hot stamping machine"],
        "is_stream": False
    }

    print("\n>>> Starting local test for node_search_embedding...")
    try:
        result = node_search_embedding(test_state)
        logger.info(f"Retrieval result summary: {result}")
        chunks = result.get("embedding_chunks", [])
        print(f"\n>>> Test finished. Retrieved {len(chunks)} results")
        
        if chunks:
            print("\n>>> Top 1 result details:")
            top1 = chunks[0]
            print(f"ID: {top1.get('id')}")
            print(f"Distance: {top1.get('distance')}")
            entity = top1.get('entity', {})
            print(f"Item Name: {entity.get('item_name')}")
            print(f"Content Preview: {entity.get('content', '')[:100]}...")
        else:
            print("\n>>> Warning: no results were retrieved. Check the Milvus data or the item_names filter.")
            
    except Exception as e:
        logger.error(f"Local test failed: {e}", exc_info=True)

