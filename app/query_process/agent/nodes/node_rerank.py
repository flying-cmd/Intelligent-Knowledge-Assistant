from app.utils.task_utils import *
from app.lm.reranker_utils import get_reranker_model
from app.core.logger import logger
import sys

# Global Rerank / TopK constants.
# Hard upper bound for dynamic TopK.
RERANK_MAX_TOPK: int = 10
# Hard lower bound for dynamic TopK.
RERANK_MIN_TOPK: int = 1
# Relative cliff threshold.
RERANK_GAP_RATIO: float = 0.25
# Absolute cliff threshold.
RERANK_GAP_ABS: float = 0.5

# Rerank node.
def step_1_merge_docs(state):
    """
    Stage 1: merge and normalize multi-source documents into one reranker-ready format.
    """
    
    # 1. Extract the inputs.
    rrf_docs = state.get("rrf_chunks") or []
    web_docs = state.get("web_search_docs") or []
    
    logger.info(f"Step 1: Starting document merge - local RRF source: {len(rrf_docs)}, web source: {len(web_docs)}")
    doc_items = []
    # 2. Process local knowledge-base documents.
    for i, doc in enumerate(rrf_docs):
        # Prefer the nested `entity` field when present, otherwise use the document itself.
        entity = doc.get("entity") if isinstance(doc, dict) and "entity" in doc else doc
        
        if not isinstance(entity, dict):
            logger.warning(f"Unexpected local-document format at index {i}: {type(entity)}")
            continue
            
        content = entity.get("content")
        if not content:
            logger.debug(f"Skipping local document with empty content (index={i}, keys={list(entity.keys())})")
            continue

        # Extract metadata.
        doc_id = entity.get("chunk_id") or entity.get("id")
        title = entity.get("title") or entity.get("item_name") or ""

        doc_items.append({
            "text": content,
            "doc_id": doc_id,
            "chunk_id": doc_id,  # Preserve the legacy field.
            "title": title,
            "url": "",
            "source": "local",
        })

    # 3. Process web-search documents.
    for i, doc in enumerate(web_docs):
        # Support multiple field names for the main text.
        text = (doc.get("snippet") or doc.get("content") or "").strip()
        url = (doc.get("url") or "").strip()
        title = (doc.get("title") or "").strip()
        
        if not text:
            logger.debug(f"Skipping empty web-search result (index={i})")
            continue
            
        doc_items.append({
            "text": text,
            "doc_id": None,
            "chunk_id": None,
            "title": title,
            "url": url,
            "source": "web",
        })

    logger.info(f"Step 1: Document merge completed, produced {len(doc_items)} normalized documents")
    return doc_items


def step_2_rerank_docs(state, doc_items):
    """
    Stage 2: rerank the normalized documents.
    """
    question = state.get("rewritten_query") or state.get("original_query") or ""

    # Skip reranking if the input is incomplete.
    if not doc_items or not question:
        logger.warning("Step 2: Skipping rerank because documents or question are missing")
        return []

    logger.info(f"Step 2: Starting rerank, document count: {len(doc_items)}")
    
    texts = [x["text"] for x in doc_items]
    try:
        reranker = get_reranker_model()

        # Build (query, passage) pairs in the order expected by the reranker.
        sentence_pairs = [[question, t] for t in texts]
        logger.info("Step 2: Computing relevance scores...")
        scores = reranker.compute_score(sentence_pairs)
        # Attach scores to docs and sort descending.
        scored_docs = []
        for item, text, score in zip(doc_items, texts, scores):
            score_val = float(score)
            scored_docs.append(
                {
                    "text": text,
                    "score": score_val,
                    "source": item.get("source") or "",
                    "chunk_id": item.get("chunk_id"),
                    "doc_id": item.get("doc_id"),
                    "url": item.get("url") or "",
                    "title": item.get("title") or "",
                }
        )
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs
    except Exception as e:
        logger.error(f"Step 2: Rerank failed: {e}", exc_info=True)
        # Fallback: keep the original order and assign zero scores.
        fallback_docs = [
            {
                "text": x.get("text"),
                "score": 0.0,
                "source": x.get("source") or "",
                "chunk_id": x.get("chunk_id"),
                "doc_id": x.get("doc_id"),
                "url": x.get("url") or "",
                "title": x.get("title") or "",
            }
            for x in doc_items
        ]
        return fallback_docs

def step_3_topk(scored_docs):
    """
    Stage 3: dynamic TopK selection with a score-cliff cutoff.
    """
    max_topk = min(RERANK_MAX_TOPK, len(scored_docs))
    min_topk = RERANK_MIN_TOPK
    gap_ratio = RERANK_GAP_RATIO
    gap_abs = RERANK_GAP_ABS

    # Detect score cliffs after the minimum guaranteed range.
    topk = max_topk
    if topk > min_topk:
        for i in range(min_topk - 1, max_topk - 1):
            s1 = scored_docs[i].get("score")
            s2 = scored_docs[i + 1].get("score")

            gap = s1 - s2
            rel = gap / (abs(s1) + 1e-6)
            if gap >= gap_abs or rel >= gap_ratio:
                logger.info(f"Step 3: Score cliff triggered at index={i} (score {s1:.4f} -> {s2:.4f}, gap={gap:.4f})")
                topk = i + 1
                break

    topk_docs = scored_docs[:topk]
    
    logger.info(f"Step 3: TopK selection finished, keeping {len(topk_docs)} documents (TopK={topk})")
    
    if topk_docs:
        preview = ", ".join([f"{d.get('chunk_id') or 'Web'}({d.get('score'):.3f})" for d in topk_docs[:3]])
        logger.debug(f"Step 3: Top 3 preview: {preview}")
        
    return topk_docs


def node_rerank(state):
  """
  Rerank node.
  """
  logger.info("---Rerank node started---")
  add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

  doc_items = step_1_merge_docs(state)
  scored_docs = step_2_rerank_docs(state, doc_items)
  topk_docs = step_3_topk(scored_docs)
  
  logger.info(f"Rerank node finished, final output count: {len(topk_docs)}")

  add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
  return {"reranked_docs": topk_docs}


if __name__ == "__main__":
    print("\n" + "="*50)
    print(">>> Starting local test for node_rerank")
    print("="*50)
    
    # 1. Mock data
    mock_rrf_chunks = [
        {"chunk_id": "local_1", "content": "RRF is a reciprocal-rank-fusion algorithm", "title": "Algorithm introduction", "score": 0.9},
        {"chunk_id": "local_2", "content": "BGE is a strong reranking model", "title": "Model introduction", "score": 0.8},
        {"chunk_id": "local_3", "content": "Unrelated test-document content", "title": "Test document", "score": 0.1}
    ]
    
    mock_web_docs = [
        {"title": "Rerank technical deep dive", "url": "http://web.com/1", "snippet": "Rerank is commonly used as the second stage in RAG systems"},
        {"title": "Unrelated webpage", "url": "http://web.com/2", "snippet": "The weather is nice today and great for going outside"}
    ]
    
    mock_state = {
        "session_id": "test_rerank_session",
        "rewritten_query": "What are RRF and Rerank?",
        "rrf_chunks": mock_rrf_chunks,
        "web_search_docs": mock_web_docs,
        "is_stream": False
    }

    try:
        result = node_rerank(mock_state)
        reranked = result.get("reranked_docs", [])
        
        print("\n" + "="*50)
        print(">>> Test result summary:")
        print(f"Total input documents: {len(mock_rrf_chunks) + len(mock_web_docs)}")
        print(f"Total output documents: {len(reranked)}")
        print("-" * 30)
        
        print("Final ranking:")
        for i, doc in enumerate(reranked, 1):
            print(f"Rank {i}: Source={doc.get('source')}, Score={doc.get('score'):.4f}, Text={doc.get('text')[:20]}...")
            
        top1_score = reranked[0].get("score")
        if top1_score > 0:
            print("\n[PASS] Rerank scoring looks valid")
        else:
            print("\n[FAIL] Rerank scoring looks invalid (all scores are zero or negative)")

        print("="*50)

    except Exception as e:
        logger.exception(f"Uncaught exception during local test: {e}")
