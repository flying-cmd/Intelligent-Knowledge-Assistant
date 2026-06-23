import os
import shutil
import uuid
from typing import List, Dict, Any
from datetime import datetime
import uvicorn
# Third-party libraries
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
# Project-internal tools, config, and clients
from app.clients.minio_utils import get_minio_client
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import (
    add_running_task,
    add_done_task,
    get_done_task_list,
    get_running_task_list,
    update_task_status,
    get_task_status,
)
from app.import_process.agent.state import get_default_state
from app.import_process.agent.main_graph import kb_import_app  # Compiled LangGraph import pipeline
from app.core.logger import logger  # Shared project logger

# Initialize the FastAPI application.
app = FastAPI(
    title="File Import Service",
    description="Web service for uploading files to the knowledge base (PDF/MD -> parsing -> splitting -> vectorization -> Milvus/KG import)"
)

# CORS middleware configuration.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all frontend origins. Restrict this in production.
    allow_credentials=True,  # Allow cookies and other auth information.
    allow_methods=["*"],  # Allow all HTTP methods.
    allow_headers=["*"],  # Allow all request headers.
)


# --------------------------
# Static page route for the import UI.
# --------------------------
@app.get("/import.html", response_class=FileResponse)
async def get_import_page():
    """Return the import frontend page: import.html."""
    # Resolve the absolute HTML path from the project root.
    html_abs_path = PROJECT_ROOT / "app/import_process/page/import.html"
    logger.info(f"Frontend import page requested, absolute path: {html_abs_path}")

    # Raise 404 if the page is missing.
    if not os.path.exists(html_abs_path):
        logger.error(f"Frontend import page not found: {html_abs_path}")
        raise HTTPException(status_code=404, detail="import.html page not found")

    # Return the HTML so the browser can render it directly.
    return FileResponse(
        path=html_abs_path,
        media_type="text/html"  # Explicit HTML media type for the browser.
    )


# --------------------------
# Background task that executes the LangGraph import pipeline.
# --------------------------
def run_graph_task(task_id: str, local_dir: str, local_file_path: str):
    """
    Run the full LangGraph import pipeline in the background.
    :param task_id: Unique task ID for a single file-import workflow
    :param local_dir: Local task directory for temporary and parsed files
    :param local_file_path: Absolute local path of the uploaded file
    """
    try:
        # 1. Mark the task as processing.
        update_task_status(task_id, "processing")
        logger.info(f"[{task_id}] Starting the LangGraph import workflow. Local file path: {local_file_path}")

        # 2. Build the initial graph state.
        init_state = get_default_state()
        init_state["task_id"] = task_id
        init_state["local_dir"] = local_dir
        init_state["local_file_path"] = local_file_path

        # 3. Stream node results as the workflow runs.
        for event in kb_import_app.stream(init_state):
            for node_name, node_result in event.items():
                logger.info(f"[{task_id}] LangGraph node completed: {node_name}")
                add_done_task(task_id, node_name)

        # 4. Mark the task as completed.
        update_task_status(task_id, "completed")
        logger.info(f"[{task_id}] LangGraph import workflow completed successfully")

    except Exception as e:
        # 5. Mark the task as failed and log the exception.
        update_task_status(task_id, "failed")
        logger.error(f"[{task_id}] LangGraph import workflow failed: {str(e)}", exc_info=True)


# --------------------------
# Core upload endpoint.
# --------------------------
@app.post("/upload", summary="File upload endpoint", description="Supports multi-file upload and automatically starts the knowledge-base import pipeline")
async def upload_files(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """
    Upload files and trigger the full knowledge-base import workflow.
    :param background_tasks: FastAPI background task manager
    :param files: Uploaded files from a multipart form
    :return: JSON response containing the generated task IDs
    """
    # 1. Build the local storage root: project_root/output/YYYYMMDD
    date_based_root_dir = os.path.join(PROJECT_ROOT / "output", datetime.now().strftime("%Y%m%d"))
    # Task IDs returned to the client.
    task_ids = []

    # 2. Process each uploaded file independently.
    for file in files:
        # Generate a globally unique task ID for the file.
        task_id = str(uuid.uuid4())
        task_ids.append(task_id)
        logger.info(f"[{task_id}] Starting upload handling. Filename: {file.filename}, content type: {file.content_type}")

        # 3. Mark the upload stage as running.
        add_running_task(task_id, "upload_file")

        # 4. Create a dedicated local directory for this task.
        task_local_dir = os.path.join(date_based_root_dir, task_id)
        os.makedirs(task_local_dir, exist_ok=True)
        # Absolute local destination for the uploaded file.
        local_file_abs_path = os.path.join(task_local_dir, file.filename)

        # 5. Save the uploaded file locally for later MinIO upload and parsing.
        with open(local_file_abs_path, "wb") as file_buffer:
            shutil.copyfileobj(file.file, file_buffer)
        logger.info(f"[{task_id}] File saved locally at: {local_file_abs_path}")

        # 6. Upload the local file to MinIO for persistent storage.
        minio_pdf_base_dir = os.getenv("MINIO_PDF_DIR", "pdf_files")
        minio_object_name = f"{minio_pdf_base_dir}/{datetime.now().strftime('%Y%m%d')}/{file.filename}"
        try:
            # Get the MinIO client instance.
            minio_client = get_minio_client()
            if minio_client is None:
                raise HTTPException(status_code=500,
                                    detail="MinIO service connection failed, please check MinIO config")
            minio_bucket_name = os.getenv("MINIO_BUCKET_NAME", "kb-import-bucket")

            # Upload the file. If the object name already exists, it will be overwritten.
            minio_client.fput_object(
                bucket_name=minio_bucket_name,
                object_name=minio_object_name,
                file_path=local_file_abs_path,
                content_type=file.content_type
            )
            logger.info(f"[{task_id}] File uploaded to MinIO successfully. Bucket: {minio_bucket_name}, object: {minio_object_name}")
        except Exception as e:
            logger.warning(f"[{task_id}] MinIO upload failed, continuing with local processing. Error: {str(e)}", exc_info=True)

        # 7. Mark the upload stage as done.
        add_done_task(task_id, "upload_file")

        # 8. Start the LangGraph workflow in the background.
        background_tasks.add_task(run_graph_task, task_id, task_local_dir, local_file_abs_path)
        logger.info(f"[{task_id}] LangGraph workflow added to background tasks and started")

    # 9. Return all generated task IDs.
    logger.info(f"Multi-file upload completed. Processed {len(files)} files, generated task IDs: {task_ids}")
    return {
        "code": 200,
        "message": f"Files uploaded successfully, total: {len(files)}",
        "task_ids": task_ids
    }


# --------------------------
# Task-status query endpoint.
# --------------------------
@app.get("/status/{task_id}", summary="Task status query", description="Query the progress and overall status of a single file by task ID")
async def get_task_progress(task_id: str):
    """
    Return the current progress of a task.
    :param task_id: Task ID returned by `/upload`
    :return: JSON response with task status, completed nodes, and running nodes
    """
    task_status_info: Dict[str, Any] = {
        "code": 200,
        "task_id": task_id,
        "status": get_task_status(task_id),  # Global task status
        "done_list": get_done_task_list(task_id),  # Completed nodes / stages
        "running_list": get_running_task_list(task_id)  # Running nodes / stages
    }
    logger.info(
        f"[{task_id}] Task status queried. Current status: {task_status_info['status']}, completed nodes: {task_status_info['done_list']}")
    return task_status_info


# Service entry point.
if __name__ == "__main__":
    """Run the service directly in a local development environment."""
    logger.info("File Import Service is starting...")
    uvicorn.run(
        app=app,
        host="127.0.0.1",  # Change to 0.0.0.0 in production if remote access is needed.
        port=8000
    )
