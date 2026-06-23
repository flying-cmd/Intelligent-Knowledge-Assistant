import sys
import os
import json
import logging
from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage

from app.core.load_prompt import load_prompt
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task
from app.clients.mongo_history_utils import get_recent_messages, save_chat_message, update_message_item_names
from app.lm.lm_utils import get_llm_client
from app.lm.embedding_utils import generate_embeddings
from app.clients.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search
from dotenv import load_dotenv, find_dotenv
from app.core.logger import logger

load_dotenv(find_dotenv())


def step_3_extract_info(query: str, history: List[Dict]) -> Dict:
    """
    Use the LLM to extract product names and rewrite the current question.
    """
    logger.info("Step 3: Starting LLM-based information extraction")
    
    client = get_llm_client(json_mode=True)
    
    # Build the conversation-history text.
    history_text = ""
    for msg in history:
        history_text += f"{msg.get('role', 'unknown')}: {msg.get('text', '')}\n"
    
    logger.info(f"Step 3: History context built, length: {len(history_text)} characters")

    try:
        prompt = load_prompt("rewritten_query_and_itemnames", history_text=history_text, query=query)
        logger.debug(f"Step 3: Prompt loaded successfully, length: {len(prompt)}")
    except Exception as e:
        logger.error(f"Step 3: Failed to load prompt: {e}")
        return {"item_names": [], "rewritten_query": query}

    messages = [
        SystemMessage(content="You are a professional customer-support assistant who is skilled at understanding user intent and extracting key information."),
        HumanMessage(content=prompt)
    ]

    try:
        logger.info("Step 3: Calling the LLM for extraction...")
        response = client.invoke(messages)
        content = response.content
        logger.debug(f"Step 3: Raw LLM response: {content}")

        # Strip surrounding Markdown code fences when present.
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "")
        
        result = json.loads(content)
        
        # Defensive normalization of the response payload.
        if "item_names" not in result:
            result["item_names"] = []
        if "rewritten_query" not in result:
            result["rewritten_query"] = query
            
        logger.info(f"Step 3: Extraction parsed successfully - item_names: {result['item_names']}, rewritten_query: {result['rewritten_query']}")
        return result

    except Exception as e:
        logger.error(f"Step 3: LLM extraction or parsing failed: {e}")
        return {"item_names": [], "rewritten_query": query}


def step_4_vectorize_and_query(item_names: List[str]) -> List[Dict]:
    """
    Vectorize extracted item names and perform Milvus hybrid retrieval.
    """
    logger.info(f"Step 4: Starting vector retrieval for target items: {item_names}")
    results = []
    
    client = get_milvus_client()
    if not client:
        logger.error("Step 4: Unable to connect to Milvus")
        return results

    collection_name = os.environ.get("ITEM_NAME_COLLECTION")
    if not collection_name:
        logger.error("Step 4: ITEM_NAME_COLLECTION was not found in the environment")
        return results

    try:
        logger.info("Step 4: Generating embeddings (dense + sparse)...")
        embeddings = generate_embeddings(item_names)
        logger.info(f"Step 4: Embeddings generated, starting Milvus search (collection: {collection_name})")

        for i, name in enumerate(item_names):
            try:
                dense_vector = embeddings.get("dense")[i]
                sparse_vector = embeddings.get("sparse")[i]

                reqs = create_hybrid_search_requests(
                    dense_vector=dense_vector,
                    sparse_vector=sparse_vector,
                    limit=5
                )

                search_res = hybrid_search(
                    client=client,
                    collection_name=collection_name,
                    reqs=reqs,
                    ranker_weights=(0.8, 0.2), 
                    limit=5,
                    norm_score=True,
                    output_fields=["item_name"]
                )

                matches = []
                if search_res and len(search_res) > 0:
                    for hit in search_res[0]:
                        entity = hit.get("entity") or {}
                        item_name = entity.get("item_name")
                        score = hit.get("distance")
                        
                        if item_name:
                            matches.append({
                                "item_name": item_name,
                                "score": score
                            })
                                logger.debug(f"Step 4: Match for '{name}': {item_name} (score: {score:.4f})")

                results.append({
                    "extracted_name": name,
                    "matches": matches
                })
                logger.info(f"Step 4: Retrieval for item '{name}' completed with {len(matches)} matches")

            except Exception as inner_e:
                logger.error(f"Step 4: Error while processing item '{name}': {inner_e}")
                results.append({"extracted_name": name, "matches": []})

    except Exception as e:
        logger.error(f"Step 4: Global vectorization or search error: {e}")

    return results


def step_5_align_item_names(query_results: List[Dict]) -> Dict:
    """
    Align extracted item names using Milvus scores and produce confirmed names
    plus candidate options.
    """
    logger.info("Step 5: Starting item-name alignment (score analysis)")
    
    confirmed_item_names = []
    options = []

    for res in query_results:
        extracted_name = res.get("extracted_name", "").strip()
        matches = res.get("matches", []) or []
        
        if not matches:
            logger.info(f"Step 5: '{extracted_name}' has no matches")
            continue

        # Sort by score descending.
        matches.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        top_matches_log = ", ".join([f"{m['item_name']}({m['score']:.3f})" for m in matches[:3]])
        logger.info(f"Step 5: Top matches for '{extracted_name}': {top_matches_log}")

        high = [m for m in matches if m.get("score", 0) > 0.85]
        mid = [m for m in matches if m.get("score", 0) >= 0.6]

        # Rule A: a single high-confidence match.
        if len(high) == 1:
            confirmed_name = high[0].get("item_name")
            confirmed_item_names.append(confirmed_name)
            logger.info(f"Step 5: Rule A matched (single high-confidence hit) -> confirmed: {confirmed_name}")
            continue

        # Rule B: multiple high-confidence matches.
        if len(high) > 1:
            picked = None
            if extracted_name:
                for m in high:
                    if m.get("item_name") == extracted_name:
                        picked = m
                        logger.info(f"Step 5: Rule B matched (exact name among high-confidence hits) -> confirmed: {picked.get('item_name')}")
                        break
            
            if not picked:
                picked = high[0]
                logger.info(f"Step 5: Rule B matched (highest score) -> confirmed: {picked.get('item_name')}")

            confirmed_item_names.append(picked.get("item_name"))
            continue

        # Rule C: no high-confidence hits, fall back to mid-confidence candidates.
        if len(mid) > 0:
            current_options = [m.get("item_name") for m in mid[:5]]
            options.extend(current_options)
            logger.info(f"Step 5: Rule C matched (mid confidence) -> candidate options: {current_options}")
            continue
        
        logger.info(f"Step 5: Rule D matched (low confidence) -> no usable match")

    result = {
        "confirmed_item_names": list(set(confirmed_item_names)),
        "options": list(set(options))
    }
    logger.info(f"Step 5: Alignment result: {result}")
    return result


def step_6_check_confirmation(state: Dict, align_result: Dict, session_id: str, history: List[Dict], rewritten_query: str) -> Dict:
    """
    Update state based on the item-name alignment result.
    """
    logger.info("Step 6: Checking confirmation state and updating state")
    
    if align_result is None:
        align_result = {}

    confirmed = align_result.get("confirmed_item_names", [])
    options = align_result.get("options", [])

    # Branch A: confirmed product names exist.
    if confirmed:
        logger.info(f"Step 6: [Branch A] Confirmed item names exist: {confirmed}")
        
        ids_to_update = []
        for msg in history:
            if not msg.get("item_names"):
                mid = msg.get("_id")
                if mid:
                    ids_to_update.append(str(mid))
        
        if ids_to_update:
            logger.info(f"Step 6: Updating item_names for {len(ids_to_update)} history messages")
            update_message_item_names(ids_to_update, confirmed)

        state["item_names"] = confirmed
        state["rewritten_query"] = rewritten_query
        if "answer" in state:
            del state["answer"]
        return state

    # Branch B: candidate item names exist.
    if options:
        logger.info(f"Step 6: [Branch B] Candidate item names exist: {options}")
        options_str = "、".join(options[:3])
        answer = f"Which of these products did you mean: {options_str}? Please specify the exact model."
        state["answer"] = answer
        state["item_names"] = []
        return state

    # Branch C: no confirmed names and no candidates.
    logger.info("Step 6: [Branch C] No confirmed names and no candidates")
    state["answer"] = "Sorry, no matching product was found. Please provide the exact model so I can look it up."
    state["item_names"] = []
    return state


def step_7_write_history(state: Dict, session_id: str, history: List[Dict], rewritten_query: str, message_id: str) -> Dict:
    """
    Write the final history records.
    """
    logger.info("Step 7: Writing session history")
    
    # If an assistant answer exists, persist it.
    if state.get("answer"):
        logger.info("Step 7: Saving assistant answer")
        save_chat_message(
            session_id=session_id,
            role="assistant",
            text=state["answer"],
            rewritten_query="",
            item_names=[]
        )

    # Update the user message with the rewritten query and item_names.
    logger.info(f"Step 7: Updating user message (ID: {message_id})")
    save_chat_message(
        session_id=session_id,
        role="user",
        text=state["original_query"],
        rewritten_query=rewritten_query,
        item_names=state.get("item_names", []),
        message_id=message_id
    )

    return state


def node_item_name_confirm(state: QueryGraphState) -> QueryGraphState:
    """
    Main node for product-name confirmation.
    """
    logger.info(">>> node_item_name_confirm: started")
    
    session_id = state["session_id"]
    original_query = state.get("original_query", "")
    is_stream = state.get("is_stream", False)

    add_running_task(session_id, "node_item_name_confirm", is_stream)

    # 1. Fetch recent history.
    history = get_recent_messages(session_id, limit=10)
    logger.info(f"Node: retrieved {len(history)} history messages")

    # 2. Save the current user message first. Step 7 may update it later.
    message_id = save_chat_message(session_id, "user", original_query, "", state.get("item_names", []))
    logger.debug(f"Node: initial user message saved, ID: {message_id}")

    # 3. Extract item names and build a rewritten query.
    extract_res = step_3_extract_info(original_query, history)
    item_names = extract_res.get("item_names", [])
    rewritten_query = extract_res.get("rewritten_query", original_query)
    
    state["rewritten_query"] = rewritten_query

    align_result = {}

    # 4. and 5. Search and align if item names were extracted.
    if len(item_names) > 0:
        query_results = step_4_vectorize_and_query(item_names)
        align_result = step_5_align_item_names(query_results)
    else:
        logger.info("Node: no item names were extracted, skipping vector retrieval")

    # 6. Update state according to the alignment result.
    state = step_6_check_confirmation(state, align_result, session_id, history, rewritten_query)

    # 7. Write the final history entries.
    final_state = step_7_write_history(state, session_id, history, rewritten_query, message_id)

    # Keep history in state for downstream nodes such as node_answer_output.
    final_state["history"] = history

    add_done_task(session_id, "node_item_name_confirm", is_stream)
    
    logger.info(f"Node: finished, final state item_names: {final_state.get('item_names')}")
    return final_state


if __name__ == "__main__":
    print("\n" + "="*50)
    print(">>> Starting local test for node_item_name_confirm")
    print("="*50)
    
    # Mock input state
    mock_state = {
        "session_id": "test_debug_session_001",
        "original_query": "How much does the HAK 180 hot stamping machine cost?",
        "is_stream": False,
        "item_names": []
    }

    try:
        result = node_item_name_confirm(mock_state)
        
        print("\n" + "="*50)
        print(">>> Test result summary:")
        print(f"Rewritten Query: {result.get('rewritten_query')}")
        print(f"Item Names: {result.get('item_names')}")
        print(f"Answer: {result.get('answer')}")
        print("="*50)

    except Exception as e:
        logger.exception(f"Uncaught exception during local test: {e}")
