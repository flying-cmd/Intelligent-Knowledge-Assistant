import sys
from app.utils.task_utils import add_running_task, add_done_task, set_task_result
from app.utils.sse_utils import push_to_session, SSEEvent
from app.query_process.agent.state import QueryGraphState
from app.core.logger import logger
from app.core.load_prompt import load_prompt
from app.lm.lm_utils import get_llm_client
from app.clients.mongo_history_utils import save_chat_message
import re

_IMAGE_BLOCK_MARKER = "[Images]"
MAX_CONTEXT_CHARS = 12000

def step_1_check_answer(state) -> bool:
  """
  Stage 1: check whether `state` already contains an answer.
  """
  answer = state.get("answer", None)
  is_stream = state.get("is_stream" )
  if answer:
    if is_stream:
      logger.info("---Step 1: Existing answer found, streaming it directly---")
      push_to_session(state["session_id"], SSEEvent.DELTA, {"delta": answer})
    else:
      set_task_result(state["session_id"], "answer", answer)
    return True
  else:
    return False

def step_2_construct_prompt(state: QueryGraphState) -> str:
  """
  Stage 2: construct the answer-generation prompt.
  """
  # 1. Gather the source data.
  original_query = state.get("original_query", "")
  rewritten_query = state.get("rewritten_query", "")
  # Prefer the rewritten query when available.
  question = rewritten_query if rewritten_query else original_query
  history = state.get("history", [])
  item_names = state.get("item_names", [])
  reranked_docs = state.get("reranked_docs") or []

  # 2. Build the reference-context string from reranked documents.
  docs = []
  used = 0
  for i, doc in enumerate(reranked_docs, start=1):
    text = (doc.get("text") or "").strip()
    if not text:
      continue
    source = doc.get("source") or ""
    chunk_id = doc.get("chunk_id")
    url = (doc.get("url") or "").strip()
    title = (doc.get("title") or "").strip()
    score = doc.get("score")

    meta_parts = [f"[{i}]"]
    if source:
      meta_parts.append(f"[{source}]")
    if chunk_id:
      meta_parts.append(f"[chunk_id={chunk_id}]")
    if url:
      meta_parts.append(f"[url={url}]")
    if score is not None:
      meta_parts.append(f"[score={float(score):.4f}]")
    if title:
      meta_parts.append(f"[title={title}]")
    doc = " ".join(meta_parts) + "\n" + text
    if used + len(doc) > MAX_CONTEXT_CHARS:
      break
    docs.append(doc)
    used += len(doc) + 2
  context_str = "\n\n".join(docs) if docs else "No reference content"


  # 3. Format the conversation history.
  history_str = ""
  if history:
    for msg in history:
      role = msg.get("role")
      text = msg.get("text")
      if role == "user" and text:
        history_str += f"User: {text}\n"
      elif role == "assistant" and text:
        history_str += f"Assistant: {text}\n"
        
      used += len(history_str) + 2
      if used > MAX_CONTEXT_CHARS:
        break
  else:
    history_str = "No conversation history"

  # 4. Format the item-name list.
  item_names_str = ", ".join(item_names) if item_names else "No specific product"

  # 5. Assemble the prompt.
  prompt = load_prompt("answer_out",
    context=context_str,
    history=history_str,
    item_names=item_names_str,
    question=question
  )

  logger.info(f"Constructed prompt: {prompt}")

  return prompt


def step_3_generate_response(state: QueryGraphState, prompt: str) -> QueryGraphState:
  """
  Stage 3: generate the answer, with optional streaming.
  """
  logger.info("---Step 3: Starting answer generation (LLM)---")
  logger.debug(f"Final prompt content: {prompt}")
  
  llm = get_llm_client()

  session_id = state.get("session_id")
  is_stream = state.get("is_stream")

  if is_stream:
    logger.info(f"Mode: streaming, session: {session_id}")
    final_text = ""
    try:
      for chunk in llm.stream(prompt):
        delta = getattr(chunk, "content", "") or ""
        if delta:
          final_text += delta
          push_to_session(session_id, SSEEvent.DELTA, {"delta": delta})
      
      logger.info(f"Streaming output finished, total length: {len(final_text)}")

    except Exception as e:
      logger.error(f"Streaming generation failed: {e}", exc_info=True)
      push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})
      
    state["answer"] = final_text
  else:
    logger.info(f"Mode: non-streaming, session: {session_id}")
    try:
      response = llm.invoke(prompt)
      content = response.content
      state["answer"] = content
      set_task_result(session_id, "answer", content)
      logger.info(f"Answer generation finished, length: {len(content)}")
    except Exception as e:
      logger.error(f"Answer generation failed: {e}", exc_info=True)
      state["answer"] = "Sorry, an error occurred while generating the answer."

  return state


def _extract_images_from_docs(docs):
    """
    Extract unique image URLs from a list of documents.
    """
    images = []
    seen = set()
    if not docs:
        return []
    md_img_pattern = re.compile(r'!\[.*?\]\((.*?)\)')

    logger.info(f"Starting image extraction, document count: {len(docs)}")

    for i, doc in enumerate(docs):
        # 1. Check direct `url` fields first, typically from web-search results.
        url = (doc.get("url") or "").strip()
        if url:
            if url.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.svg')):
                if url not in seen:
                    logger.debug(f"Document [{i}] found image URL in field: {url}")
                    seen.add(url)
                    images.append(url)

        # 2. Extract Markdown image syntax from text fields.
        text = (doc.get("text") or "").strip()
        if text:
            matches = md_img_pattern.findall(text)
            for img_url in matches:
                img_url = img_url.strip()
                if img_url and img_url not in seen:
                    logger.debug(f"Document [{i}] found Markdown image in text: {img_url}")
                    seen.add(img_url)
                    images.append(img_url)

    logger.info(f"Image extraction finished. Found {len(images)} unique images: {images}")
    return images


def step_4_write_history(state: QueryGraphState, image_urls = None) -> QueryGraphState:
  """
  Stage 4: write the current assistant answer into MongoDB history.
  """
  session_id = state.get("session_id", "default")
  answer = (state.get("answer") or "").strip()
  item_names = state.get("item_names") or []

  try:
    if answer:
       save_chat_message(
        session_id=session_id,
        role="assistant",
        text=answer,
        rewritten_query="",
        item_names=item_names,
        image_urls=image_urls,
        message_id=None
      )
  except Exception as e:
    logger.error(f"Failed to write MongoDB history: {e}")

  return state


def node_answer_output(state: QueryGraphState) -> QueryGraphState:
  """
  Final answer-output node.
  """
  logger.info("---node_answer_output (answer generation) node started---")
  add_running_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
  
  answer_exists = step_1_check_answer(state)
  
  if not answer_exists:
    prompt = step_2_construct_prompt(state)
    state["prompt"] = prompt

    step_3_generate_response(state, prompt)

  # Extract image URLs for history storage and frontend rendering.
  image_urls = _extract_images_from_docs(state.get("reranked_docs") or [])

  if state.get("answer"):
    logger.info("---Writing MongoDB history---")
    step_4_write_history(state, image_urls=image_urls)

  add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
  
  # Final SSE event to ensure the frontend can render images and close cleanly.
  logger.info(f"---Sending final event--- image_urls={image_urls}")
  if state.get("is_stream"):
    push_to_session(
        state['session_id'],
        SSEEvent.FINAL,
        {
            "answer": state["answer"],
            "status": "completed",
            "image_urls": image_urls
        }
    )
  
  logger.info("---node_answer_output finished---")
  return state


if __name__ == "__main__":
    print("\n" + "="*50)
    print(">>> Starting local test for node_answer_output")
    print("="*50)
    
    # 1. Build mock data.
    mock_reranked_docs = [
        {
            "chunk_id": "local_101",
            "source": "local",
            "title": "HAK 180 hot-stamping-machine operation manual v2.pdf",
            "score": 0.95,
            "text": """
            The HAK 180 hot stamping machine control panel is located on the front of the machine.
            After powering it on, set the temperature first. The recommended default is about 110 C.
            Refer to the image below for the panel layout:
            ![Control panel layout](http://local-server/images/panel_view.jpg)
            
            For partial hot stamping, adjust the side knob.
            ![Side knob detail](http://local-server/images/knob_detail.png)
            """
        },
        {
            "chunk_id": None,
            "source": "web",
            "title": "HAK 180 troubleshooting - official site",
            "score": 0.88,
            "url": "http://example.com/hak180_troubleshooting.jpeg",
            "text": "If the machine does not heat up, check whether the fuse has blown..."
        },
        {
            "chunk_id": "local_102",
            "source": "local",
            "title": "Safety precautions",
            "score": 0.82,
            "text": "Wear heat-resistant gloves during operation to avoid burns."
        }
    ]

    # Mock history
    mock_history = [
        {"role": "user", "text": "Hello, how do I use this machine?"},
        {"role": "assistant", "text": "Hello! Which machine model are you asking about?"},
        {"role": "user", "text": "HAK 180 hot stamping machine"}
    ]

    # Mock input state
    mock_state = {
        "session_id": "test_answer_session_001",
        "original_query": "How do I operate the HAK 180 hot stamping machine?",
        "rewritten_query": "What are the detailed operating steps and panel settings for the HAK 180 hot stamping machine?",
        "item_names": ["HAK 180 hot stamping machine"],
        "history": mock_history,
        "reranked_docs": mock_reranked_docs,
        "is_stream": False,
        "answer": None
    }

    try:
        result = node_answer_output(mock_state)
        
        print("\n" + "="*50)
        print(">>> Test result summary:")
        
        if "prompt" in result:
            print(f"[PASS] Prompt built successfully (length: {len(result['prompt'])})")
        else:
            print("[FAIL] Prompt was not built")

        answer = result.get("answer")
        if answer and len(answer) > 10:
            print(f"[PASS] Answer generated successfully (length: {len(answer)})")
            print(f"Answer preview: {answer[:50]}...")
        else:
            print(f"[WARN] Answer generation may have failed (content: {answer})")

        print("\n[INFO] Check the logs above for 'Image extraction finished' and these URLs:")
        print(" - http://local-server/images/panel_view.jpg")
        print(" - http://local-server/images/knob_detail.png")
        print(" - http://example.com/hak180_troubleshooting.jpeg")

        print("="*50)

    except Exception as e:
        logger.exception(f"Uncaught exception during local test: {e}")
