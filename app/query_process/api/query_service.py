from pathlib import Path
import uuid
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

from app.utils.task_utils import *
from app.utils.sse_utils import create_sse_queue, SSEEvent, sse_generator
from app.clients.mongo_history_utils import *
from app.query_process.agent.main_graph import query_app

# Future import graph entry point.
#from app.query_process.main_graph import query_app


# Define the FastAPI application.
app = FastAPI(title="query service",description="Shopkeeper knowledge-base query service")
# CORS configuration.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Return the chat.html page.
@app.get("/chat.html")
async def chat():
    # Move from api/ to query_process/
    current_dir_parent_path = Path(__file__).absolute().parent.parent
    # chat.html location
    chat_html_path = current_dir_parent_path / "page" / "chat.html"
    # Raise 404 if the page cannot be found.
    if not chat_html_path.exists():
        raise HTTPException(status_code=404, detail=f"Page not found: {chat_html_path}")
    return FileResponse(chat_html_path)

# Request schema.
class QueryRequest(BaseModel):
    """Query request payload."""
    query: str = Field(..., description="Query text")
    session_id: str = Field(None, description="Session ID")
    is_stream: bool = Field(False, description="Whether to return a stream")



# Health check endpoint.
@app.get("/health")
async def health():
    """
    Check whether the service is healthy.
    """
    return {"ok": True}


# Query graph runner.
def run_query_graph(session_id: str, user_query: str, is_stream: bool = True):
    print(f"Starting query graph processing... {session_id} {user_query} {is_stream}")

    default_state = {"original_query": user_query, "session_id": session_id, "is_stream": is_stream}
    try:
        # Run the graph.
        query_app.invoke(default_state)
        # Mark the task as completed.
        update_task_status(session_id, TASK_STATUS_COMPLETED, is_stream)
    except Exception as e:
        print(f"Query graph execution error: {e}")
        update_task_status(session_id, TASK_STATUS_FAILED, is_stream)
        if is_stream:
            push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})



@app.post("/query")
async def query(background_tasks: BackgroundTasks, request: QueryRequest):
    """
    1. Parse parameters
    2. Update task status
    3. Run the graph
    4. Return the result
    :param background_tasks:
    :param request:
    :return:
    """
    user_query = request.query
    session_id = request.session_id if request.session_id else str(uuid.uuid4())

    # Handle streaming mode.
    is_stream = request.is_stream
    if is_stream:
        # Create the per-session queue used for streaming results.
        create_sse_queue(session_id)
    # Mark the task as processing.
    update_task_status(session_id, TASK_STATUS_PROCESSING,is_stream)

    print("Starting processing. is_stream:", is_stream, f"other params: {user_query}, session_id: {session_id}")

    if is_stream:
        # Stream mode: run in the background and push incremental events.
        background_tasks.add_task(run_query_graph, session_id,user_query,is_stream)
        print("Streaming task launched...")
        return {
            "message":"Result is being processed...",
            "session_id":session_id
        }
    else:
        # Synchronous mode.
        run_query_graph(session_id, user_query, is_stream)
        answer = get_task_result(session_id,"answer","")
        return {
            "message":"Processing completed",
            "session_id":session_id,
            "answer":answer,
            "done_list":[]
        }



@app.get("/stream/{session_id}")
async def stream(session_id: str, request: Request):
    print("Opening /stream endpoint...")
    """
    Return SSE events in real time.
    """
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/history/{session_id}")
async def history(session_id: str, limit: int = 50):
    """
    Return the conversation history for the current session.
    """
    try:
        records = get_recent_messages(session_id, limit=limit)
        items = []
        for r in records:
            items.append({
                "_id": str(r.get("_id")) if r.get("_id") is not None else "",
                "session_id": r.get("session_id", ""),
                "role": r.get("role", ""),
                "text": r.get("text", ""),
                "rewritten_query": r.get("rewritten_query", ""),
                "item_names": r.get("item_names", []),
                "ts": r.get("ts")
            })
        return {"session_id": session_id, "items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"history error: {e}")


@app.delete("/history/{session_id}")
async def clear_chat_history(session_id: str):
    count = clear_history(session_id)
    return {"message": "History cleared", "deleted_count": count}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
