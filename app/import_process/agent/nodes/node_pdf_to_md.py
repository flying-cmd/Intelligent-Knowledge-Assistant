# Standard library imports
import os
import sys
import time
import requests
import zipfile
import shutil
from pathlib import Path

# Project imports
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.format_utils import format_state
from app.utils.task_utils import add_running_task, add_done_task
from app.conf.mineru_config import mineru_config
from app.core.logger import logger  # Shared logger

# Cached MinerU configuration
MINERU_BASE_URL = mineru_config.base_url
MINERU_API_TOKEN = mineru_config.api_key


def step_1_validate_paths(state):
    """
    Step 1: validate the PDF path and output directory.

    Responsibilities:
    - Validate required parameters
    - Check that the PDF file is valid
    - Create the output directory automatically when needed

    Returns:
        A tuple of the validated PDF Path and output Path.

    Raises:
        ValueError: required parameters are missing.
        FileNotFoundError: the PDF path is invalid.
    """
    log_prefix = "[step_1_validate_paths] "
    pdf_path = state.get("pdf_path", "").strip()
    local_dir = state.get("local_dir", "").strip()

    if not pdf_path:
        raise ValueError(f"{log_prefix}Missing valid workflow parameter: pdf_path. Current value: {repr(pdf_path)}")
    if not local_dir:
        raise ValueError(f"{log_prefix}Missing valid workflow parameter: local_dir. Current value: {repr(local_dir)}")

    pdf_path_obj = Path(pdf_path)
    output_dir_obj = Path(local_dir)

    if not pdf_path_obj.exists():
        raise FileNotFoundError(f"{log_prefix}PDF file does not exist: {pdf_path_obj.absolute()}")
    if not pdf_path_obj.is_file():
        raise FileNotFoundError(f"{log_prefix}The specified PDF path is not a file: {pdf_path_obj.absolute()}")

    if not output_dir_obj.exists():
        logger.info(f"{log_prefix}Output directory does not exist and will be created: {output_dir_obj.absolute()}")
        output_dir_obj.mkdir(parents=True, exist_ok=True)

    return pdf_path_obj, output_dir_obj


def step_2_upload_and_poll(pdf_path_obj: Path, output_dir_obj: Path):
    """
    Step 2: upload the PDF to MinerU and poll until parsing completes.

    Flow:
    - Validate configuration
    - Request an upload URL
    - Upload the file, with retry logic
    - Poll the task until success, failure, or timeout

    Returns:
        The download URL for the result ZIP archive.

    Raises:
        ValueError: configuration is missing.
        RuntimeError: request or upload failed.
        TimeoutError: the task timed out.
    """
    if not MINERU_BASE_URL or not MINERU_API_TOKEN:
        raise ValueError("MinerU configuration is missing. Set MINERU_BASE_URL and MINERU_API_TOKEN in .env.")
    logger.info(f"[Config check] MinerU configuration loaded successfully. Processing file: {pdf_path_obj.name}")

    request_headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MINERU_API_TOKEN}"
    }

    url_get_upload = f"{MINERU_BASE_URL}/file-urls/batch"
    req_data = {
        "files": [{"name": pdf_path_obj.name}],
        "model_version": "vlm"  # Recommended parsing model
    }
    logger.debug(f"[Get upload URL] Calling API: {url_get_upload}, request payload: {req_data}")
    resp = requests.post(url=url_get_upload, headers=request_headers, json=req_data, timeout=30)

    if resp.status_code != 200:
        raise RuntimeError(f"[Get upload URL] Request failed. Status code: {resp.status_code}, response: {resp.text}")

    resp_data = resp.json()
    if resp_data["code"] != 0:
        raise RuntimeError(f"[Get upload URL] API returned a business error: {resp_data}")

    signed_url = resp_data["data"]["file_urls"][0]
    batch_id = resp_data["data"]["batch_id"]
    logger.info(f"[Get upload URL] Success. batch_id={batch_id}, upload URL generated")

    logger.info(f"[File upload] Reading PDF file: {pdf_path_obj.name}")
    with open(pdf_path_obj, "rb") as f:
        file_data = f.read()

    upload_session = requests.Session()
    upload_session.trust_env = False

    try:
        put_resp = upload_session.put(url=signed_url, data=file_data, timeout=60)
        if put_resp.status_code != 200:
            logger.warning(
                f"[File upload] Initial upload failed (status code: {put_resp.status_code}). Retrying with PDF content type."
            )
            pdf_headers = {"Content-Type": "application/pdf"}
            put_resp = upload_session.put(url=signed_url, data=file_data, headers=pdf_headers, timeout=60)
            if put_resp.status_code != 200:
                raise RuntimeError(
                    f"[File upload] Retry still failed. Status code: {put_resp.status_code}, response: {put_resp.text}"
                )
        logger.info(f"[File upload] Success. File {pdf_path_obj.name} has been uploaded")
    except Exception as e:
        raise RuntimeError(f"[File upload] Upload failed due to a network error: {str(e)}")
    finally:
        upload_session.close()

    poll_url = f"{MINERU_BASE_URL}/extract-results/batch/{batch_id}"
    start_time = time.time()
    timeout_seconds = 600  # Up to 10 minutes
    poll_interval = 3
    logger.info(f"[Task polling] Monitoring task status. batch_id={batch_id}, timeout={timeout_seconds}s")

    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout_seconds:
            raise TimeoutError(f"[Task polling] Timed out after {int(timeout_seconds)} seconds. batch_id={batch_id}")

        try:
            poll_resp = requests.get(url=poll_url, headers=request_headers, timeout=10)
        except Exception as e:
            logger.warning(f"[Task polling] Request failed. Retrying in {poll_interval} seconds: {str(e)}")
            time.sleep(poll_interval)
            continue

        if poll_resp.status_code != 200:
            if 500 <= poll_resp.status_code < 600:
                logger.warning(
                    f"[Task polling] Server is busy (status code: {poll_resp.status_code}). Retrying in {poll_interval} seconds."
                )
                time.sleep(poll_interval)
                continue
            raise RuntimeError(
                f"[Task polling] HTTP request failed. Status code: {poll_resp.status_code}, response: {poll_resp.text}"
            )

        poll_data = poll_resp.json()
        if poll_data["code"] != 0:
            raise RuntimeError(f"[Task polling] API returned a business error: {poll_data}")

        extract_results = poll_data["data"]["extract_result"]
        if not extract_results:
            logger.debug(f"[Task polling] Result is still empty after {int(elapsed_time)}s. Waiting...")
            time.sleep(poll_interval)
            continue

        result_item = extract_results[0]
        state_status = result_item["state"]
        if state_status == "done":
            logger.info(f"[Task polling] Parsing complete. Total elapsed time: {int(elapsed_time)}s, batch_id={batch_id}")
            full_zip_url = result_item.get("full_zip_url")
            if not full_zip_url:
                raise RuntimeError(f"[Task polling] Task finished but no ZIP URL was returned. batch_id={batch_id}")
            logger.info(f"[Task polling] Result ZIP download URL: {full_zip_url}...")
            return full_zip_url
        if state_status == "failed":
            err_msg = result_item.get("err_msg", "Unknown error")
            raise RuntimeError(f"[Task polling] Parsing task failed. batch_id={batch_id}, error: {err_msg}")

        logger.debug(
            f"[Task polling] Processing ({int(elapsed_time)}s elapsed), status: {state_status}, refresh interval: {poll_interval}s",
            end="\r"
        )
        time.sleep(poll_interval)


def step_3_download_and_extract(zip_url: str, output_dir_obj: Path, pdf_stem: str) -> str:
    """
    Step 3: download the MinerU ZIP result, extract it, and select the target Markdown file.

    Flow:
    - Download the ZIP
    - Clear the old extraction directory
    - Extract the ZIP
    - Find the target Markdown file by priority
    - Rename it to match the PDF stem when needed

    Returns:
        The absolute path to the final Markdown file.

    Raises:
        RuntimeError: the ZIP download failed.
        FileNotFoundError: no Markdown file was found.
    """
    logger.info(f"===== Processing MinerU results for [{pdf_stem}] =====")

    logger.info(f"[Step 1/4] Downloading ZIP archive: {zip_url}...")
    resp = requests.get(zip_url, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"[Step 1/4] Failed to download ZIP archive. HTTP status: {resp.status_code}")

    zip_save_path = output_dir_obj / f"{pdf_stem}_result.zip"
    with open(zip_save_path, "wb") as f:
        f.write(resp.content)
    logger.info(f"[Step 1/4] ZIP archive downloaded successfully: {zip_save_path}")

    logger.info("[Step 2/4] Extracting ZIP archive...")
    extract_target_dir = output_dir_obj / pdf_stem

    if extract_target_dir.exists():
        try:
            shutil.rmtree(extract_target_dir)
            logger.info(f"[Step 2/4] Removed previous extraction directory: {extract_target_dir}")
        except Exception as e:
            logger.warning(f"[Step 2/4] Failed to clean old directory. Extraction may still succeed: {str(e)}")

    extract_target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_save_path, "r") as zip_file_obj:
        zip_file_obj.extractall(extract_target_dir)
    logger.info(f"[Step 2/4] ZIP archive extracted: {extract_target_dir}")

    logger.info("[Step 3/4] Searching for Markdown files in the extracted directory...")
    md_file_list = list(extract_target_dir.rglob("*.md"))
    if not md_file_list:
        raise FileNotFoundError(f"[Step 3/4] No .md file was found in: {extract_target_dir}")
    logger.info(f"[Step 3/4] Found {len(md_file_list)} Markdown files. Selecting target by priority.")

    target_md_file = None
    for md_file in md_file_list:
        if md_file.stem == pdf_stem:
            target_md_file = md_file
            logger.info(f"[Step 4/4] Priority 1 match found: Markdown file matching the PDF name: {target_md_file.name}")
            break

    if not target_md_file:
        for md_file in md_file_list:
            if md_file.name.lower() == "full.md":
                target_md_file = md_file
                logger.info(f"[Step 4/4] Priority 2 match found: default MinerU file {target_md_file.name}")
                break

    if not target_md_file:
        target_md_file = md_file_list[0]
        logger.info(f"[Step 4/4] No higher-priority match found. Falling back to the first Markdown file: {target_md_file.name}")

    if target_md_file.stem != pdf_stem:
        logger.info(f"[Step 4/4] Renaming Markdown file to match the PDF stem: {pdf_stem}.md")
        new_md_path = target_md_file.with_name(f"{pdf_stem}.md")
        try:
            target_md_file.rename(new_md_path)
            target_md_file = new_md_path
            logger.info(f"[Step 4/4] Markdown file renamed successfully: {pdf_stem}.md")
        except OSError as e:
            logger.warning(f"[Step 4/4] Failed to rename Markdown file. Continuing with original name: {str(e)}")

    final_md_path = str(target_md_file.absolute())
    logger.info(f"===== Finished processing [{pdf_stem}]. Final Markdown path: {final_md_path} =====")
    return final_md_path


def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    LangGraph node: convert a PDF into Markdown.

    Flow:
    - Validate paths
    - Upload to MinerU and wait for parsing
    - Download and extract the result
    - Read the Markdown content back into the workflow state
    """
    func_name = sys._getframe().f_code.co_name

    logger.debug(f"[{func_name}] Node started.\nCurrent workflow state: {format_state(state)}")
    add_running_task(state["task_id"], func_name)

    try:
        pdf_path_obj, output_dir_obj = step_1_validate_paths(state)
        zip_url = step_2_upload_and_poll(pdf_path_obj, output_dir_obj)
        md_path = step_3_download_and_extract(zip_url, output_dir_obj, pdf_path_obj.stem)

        state["md_path"] = md_path
        logger.info(f"[{func_name}] Markdown file created successfully: {md_path}")

        try:
            with open(md_path, "r", encoding="utf-8") as f:
                state["md_content"] = f.read()
            logger.debug(f"[{func_name}] Markdown content loaded successfully. Length: {len(state['md_content'])} characters")
        except Exception as e:
            logger.error(f"[{func_name}] Failed to read Markdown content: {str(e)}")

        logger.info(f"[{func_name}] Node finished. Updated workflow keys: {list(state.keys())}")

    except Exception as e:
        logger.error(f"[{func_name}] PDF-to-Markdown flow failed: {str(e)}", exc_info=True)
        raise
    finally:
        add_done_task(state["task_id"], func_name)
        logger.debug(f"[{func_name}] Node finished.\nUpdated workflow state: {format_state(state)}")

    return state


if __name__ == "__main__":
    logger.info("===== Starting node_pdf_to_md local test =====")

    from app.utils.path_util import PROJECT_ROOT

    logger.info(f"Detected project root: {PROJECT_ROOT}")

    doc_dir = Path(PROJECT_ROOT) / "doc"
    test_pdf_obj = next(doc_dir.glob("*.pdf"), None)
    if not test_pdf_obj:
        raise FileNotFoundError(f"No PDF file was found in the doc directory: {doc_dir}")

    test_state = create_default_state(
        task_id="test_pdf2md_task_001",
        pdf_path=str(test_pdf_obj),
        local_dir=os.path.join(PROJECT_ROOT, "output")
    )

    node_pdf_to_md(test_state)

    logger.info("===== Finished node_pdf_to_md local test =====")
