import sys
from typing import List, Dict, Any
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger


# RRF node.
def _as_entity_list(state_list) -> List[Dict[str, Any]]:
    """
    Normalize upstream results into a list of entity dictionaries.
    """
    out: List[Dict[str, Any]] = []
    for doc in (state_list or []):
        if not doc:
            continue
        
        final_ent = {}
        
        # Case A: a Pymilvus hit-like object with `entity` and `id`.
        if hasattr(doc, "entity") and hasattr(doc, "id"):
            entity_content = doc.entity
            if hasattr(entity_content, "to_dict"):
                 final_ent = entity_content.to_dict()
            elif isinstance(entity_content, dict):
                 final_ent = entity_content.copy()
            else:
                 try:
                     final_ent = dict(entity_content)
                 except:
                     pass
            
            if "id" not in final_ent and "chunk_id" not in final_ent:
                final_ent["id"] = doc.id
            
            if hasattr(doc, "distance"):
                final_ent["score"] = doc.distance

        # Case B: a dict-like result.
        elif isinstance(doc, dict):
             if "entity" in doc:
                 ent = doc["entity"]
                 if isinstance(ent, dict):
                     final_ent = ent.copy()
                 if "id" in doc and "id" not in final_ent:
                     final_ent["id"] = doc["id"]
                 if "distance" in doc:
                     final_ent["score"] = doc["distance"]
             else:
                 final_ent = doc

        # Case C: other objects that expose a `.get` method.
        elif hasattr(doc, "get"):
             ent = doc.get("entity") or doc
             if isinstance(ent, dict):
                 final_ent = ent
        
        if final_ent and isinstance(final_ent, dict):
            out.append(final_ent)
            
    return out


def reciprocal_rank_fusion(
        source_weights: list,
        k: int = 60,
        max_results: int = None,
) -> List[tuple]:
    """
    Weighted Reciprocal Rank Fusion implementation.
    """
    score_map = {}
    chunk_map = {}

    # 1. Accumulate weighted RRF scores from every source.
    for docs, weight in source_weights:
        for rank, item in enumerate(docs, start=1):
            chunk_id = item.get("chunk_id") or item.get("id")
            
            if not chunk_id:
                logger.warning(
                    f"RRF warning: item is missing chunk_id/id: {list(item.keys()) if isinstance(item, dict) else item}")
                continue

            score_map[chunk_id] = score_map.get(chunk_id, 0.0) + weight * (1.0 / (k + rank))
            chunk_map.setdefault(chunk_id, item)

    # 2. Convert the score map to a sorted list.
    merged = []
    for chunk_id, score in score_map.items():
        doc_item = chunk_map[chunk_id]
        merged.append((doc_item, score))
    
    merged.sort(key=lambda x: x[1], reverse=True)
    
    # 3. Trim the results if a limit is provided.
    if max_results is not None:
        merged = merged[:max_results]
        
    return merged


def node_rrf(state):
    """
    RRF (Reciprocal Rank Fusion) node.
    """
    logger.info("---RRF node started---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    embedding_chunks = _as_entity_list(state.get("embedding_chunks"))
    hyde_embedding_chunks = _as_entity_list(state.get("hyde_embedding_chunks"))

    logger.info(f"RRF input stats: embedding source={len(embedding_chunks)}, HyDE source={len(hyde_embedding_chunks)}")
    
    if embedding_chunks:
        logger.debug(f"Embedding-source chunk_ids (first 5): {[c.get('chunk_id') for c in embedding_chunks[:5]]}")
    if hyde_embedding_chunks:
        logger.debug(f"HyDE-source chunk_ids (first 5): {[c.get('chunk_id') for c in hyde_embedding_chunks[:5]]}")

    source_weights = [
        (embedding_chunks, 1.0),
        (hyde_embedding_chunks, 1.0)
    ]

    rrf_res = reciprocal_rank_fusion(source_weights, k=60, max_results=10)

    rrf_chunks = [doc for doc, score in rrf_res]
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))

    return {"rrf_chunks": rrf_chunks}


if __name__ == "__main__":
    print("\n" + "="*50)
    print(">>> Starting local test for node_rrf")
    print("="*50)

    # 1. Build mock data
    mock_embedding_chunks = [
        {
            "id": "doc_1", 
            "pk": "pk_1", 
            "file_title": "operation-manual-v1.pdf", 
            "item_name": "HAK 180 hot stamping machine", 
            "content": "Content 1: Turn on the power switch...", 
            "score": 0.9
        },
        {
            "id": "doc_2", 
            "pk": "pk_2", 
            "file_title": "maintenance-guide.pdf", 
            "item_name": "HAK 180 hot stamping machine", 
            "content": "Content 2: If a fault occurs, contact support...", 
            "score": 0.8
        },
        {
            "id": "doc_3", 
            "pk": "pk_3", 
            "file_title": "parameters.xlsx", 
            "item_name": "HAK 180 hot stamping machine", 
            "content": "Content 3: Voltage 220V...", 
            "score": 0.7
        }
    ]
    
    # Simulated HyDE results with a different order plus a new doc_4.
    mock_hyde_chunks = [
        {
            "id": "doc_3", 
            "pk": "pk_3", 
            "file_title": "parameters.xlsx", 
            "item_name": "HAK 180 hot stamping machine", 
            "content": "Content 3: Voltage 220V...", 
            "score": 0.85
        }, 
        {
            "id": "doc_1", 
            "pk": "pk_1", 
            "file_title": "operation-manual-v1.pdf", 
            "item_name": "HAK 180 hot stamping machine", 
            "content": "Content 1: Turn on the power switch...", 
            "score": 0.82
        }, 
        {
            "id": "doc_4", 
            "pk": "pk_4", 
            "file_title": "safety-notes.docx", 
            "item_name": "HAK 180 hot stamping machine", 
            "content": "Content 4: Wear gloves during operation...", 
            "score": 0.75
        }
    ]

    mock_state = {
        "session_id": "test_rrf_session",
        "is_stream": False,
        "embedding_chunks": mock_embedding_chunks,
        "hyde_embedding_chunks": mock_hyde_chunks
    }

    try:
        result = node_rrf(mock_state)
        
        rrf_chunks = result.get("rrf_chunks", [])
        print("\n" + "="*50)
        print(">>> Test result summary:")
        print(f"Input count: embedding={len(mock_embedding_chunks)}, HyDE={len(mock_hyde_chunks)}")
        print(f"Output count: {len(rrf_chunks)}")
        print("-" * 30)
        
        print("Final ranking:")
        for i, doc in enumerate(rrf_chunks, 1):
            doc_id = doc.get('chunk_id') or doc.get('id')
            print(f"Rank {i}: ID={doc_id}, Title={doc.get('file_title')}, Content={doc.get('content')[:20]}...")

        ids = [d.get("id") or d.get("chunk_id") for d in rrf_chunks]
        
        if "doc_1" in ids and "doc_3" in ids:
            print("\n[PASS] Overlapping documents (doc_1, doc_3) were preserved after fusion")
        else:
            print("\n[FAIL] Overlapping documents were lost")
            
        if len(ids) == 4:
            print("[PASS] Union count is correct (3 + 3 - 2 overlaps = 4)")
        else:
            print(f"[FAIL] Incorrect union count: expected 4, got {len(ids)}")
            
        print("="*50)

    except Exception as e:
        logger.exception(f"Uncaught exception during local test: {e}")
