import os
import re
import sys
import base64
from pathlib import Path
from typing import Dict, List, Tuple
from collections import deque

# MinIO dependencies
from minio import Minio
from minio.deleteobjects import DeleteObject

from app.clients.minio_utils import get_minio_client
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task
from app.lm.lm_utils import get_llm_client
from langchain.messages import HumanMessage
from langchain_core.exceptions import LangChainException
from app.conf.minio_config import minio_config
from app.conf.lm_config import lm_config
from app.core.logger import logger
from app.utils.rate_limit_utils import apply_api_rate_limit
from app.core.load_prompt import load_prompt

# Supported image extensions for MinIO upload and Markdown processing.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


def step_1_get_content(state: ImportGraphState) -> Tuple[str, Path, Path]:
    """
    Step 1: initialize the core Markdown data.

    Returns:
        (Markdown content, Markdown file path, images directory path)
    """
    md_file_path = state["md_path"]
    if not md_file_path:
        raise FileNotFoundError(f"No valid Markdown path was found in workflow state: {state['md_path']}")

    path_obj = Path(md_file_path)
    if not state["md_content"]:
        with open(path_obj, "r", encoding="utf-8") as f:
            md_content = f.read()
        logger.debug(f"Loaded Markdown content from file, size: {len(md_content)} characters")
    else:
        md_content = state["md_content"]
        logger.debug(f"Loaded Markdown content from workflow state, size: {len(md_content)} characters")

    images_dir = path_obj.parent / "images"
    return md_content, path_obj, images_dir


def is_supported_image(filename: str) -> bool:
    """
    Return True when the file extension is in the supported image set.
    """
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS


def find_image_in_md(md_content: str, image_filename: str, context_len: int = 100) -> List[Tuple[str, str]]:
    """
    Find all references to a given image file in Markdown and capture surrounding context.
    """
    pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_filename) + r".*?\)")
    results = []

    for match in pattern.finditer(md_content):
        start, end = match.span()
        pre_text = md_content[max(0, start - context_len):start]
        post_text = md_content[end:min(len(md_content), end + context_len)]
        logger.debug(f"Found Markdown reference for image [{image_filename}], preceding text: {pre_text.strip()}")
        logger.debug(f"Found Markdown reference for image [{image_filename}], following text: {post_text.strip()}")
        results.append((pre_text, post_text))

    if not results:
        logger.debug(f"No Markdown reference found for image [{image_filename}]")
    return results


def step_2_scan_images(md_content: str, images_dir: Path) -> List[Tuple[str, str, Tuple[str, str]]]:
    """
    Step 2: scan the images directory and keep only supported images that are referenced in the Markdown.
    """
    targets = []
    for image_file in os.listdir(images_dir):
        if not is_supported_image(image_file):
            logger.debug(f"Unsupported image format, skipping: {image_file}")
            continue

        img_path = str(images_dir / image_file)
        context_list = find_image_in_md(md_content, image_file)
        if not context_list:
            logger.warning(f"Image is not referenced in the Markdown and will be skipped: {image_file}")
            continue

        targets.append((image_file, img_path, context_list[0]))
        logger.info(f"Image added to processing queue: {image_file}")

    logger.info(f"Image scan completed. Total images selected for processing: {len(targets)}")
    return targets


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode a local image file as a base64 string for multimodal LLM input.
    """
    with open(image_path, "rb") as img_file:
        base64_str = base64.b64encode(img_file.read()).decode("utf-8")
    logger.debug(f"Base64 encoding completed for image {image_path}, encoded length: {len(base64_str)}")
    return base64_str


def summarize_image(image_path: str, root_folder: str, image_content: Tuple[str, str]) -> str:
    """
    Generate a short image summary with the multimodal LLM.

    The summary is used as Markdown alt text.
    """
    base64_image = encode_image_to_base64(image_path)
    try:
        lvm_client = get_llm_client(model=lm_config.lv_model)

        prompt_text = load_prompt(
            name="image_summary",
            root_folder=root_folder,
            image_content=image_content,
        )

        messages = [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": prompt_text,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ]
            )
        ]

        response = lvm_client.invoke(messages)
        summary = response.content.strip().replace("\n", "")
        logger.info(f"Image summary generated successfully for {image_path}: {summary}")
        return summary

    except LangChainException as e:
        logger.error(f"Image summary generation failed due to a LangChain error for {image_path}: {str(e)}")
        return "Image description"
    except Exception as e:
        logger.error(f"Image summary generation failed due to a system error for {image_path}: {str(e)}")
        return "Image description"


def step_3_generate_summaries(
    doc_stem: str,
    targets: List[Tuple[str, str, Tuple[str, str]]],
    requests_per_minute: int = 9,
) -> Dict[str, str]:
    """
    Step 3: generate summaries for all selected images with API rate limiting.
    """
    summaries = {}
    request_times = deque()

    for img_file, image_path, context in targets:
        apply_api_rate_limit(request_times, requests_per_minute, window_seconds=60)
        logger.debug(f"Generating image summary for: {image_path}")
        summaries[img_file] = summarize_image(image_path, root_folder=doc_stem, image_content=context)

    logger.info(f"Image summary generation completed for {len(summaries)} images")
    return summaries


def clean_minio_directory(minio_client: Minio, prefix: str) -> None:
    """
    Idempotently remove existing objects under the given MinIO prefix.
    """
    try:
        objects_to_delete = minio_client.list_objects(
            bucket_name=minio_config.bucket_name,
            prefix=prefix,
            recursive=True,
        )
        delete_list = [DeleteObject(obj.object_name) for obj in objects_to_delete]

        if delete_list:
            logger.info(f"Cleaning old MinIO files under {prefix}. Objects to delete: {len(delete_list)}")
            errors = minio_client.remove_objects(minio_config.bucket_name, delete_list)
            for error in errors:
                logger.error(f"Failed to delete MinIO object: {error}")
        else:
            logger.debug(f"No old MinIO objects found under {prefix}")
    except Exception as e:
        logger.error(f"Failed to clean MinIO directory {prefix}: {str(e)}")


def upload_images_batch(
    minio_client: Minio,
    upload_dir: str,
    targets: List[Tuple[str, str, Tuple[str, str]]],
) -> Dict[str, str]:
    """
    Upload all processed images to MinIO and return a filename-to-URL mapping.
    """
    urls = {}
    for img_file, img_path, _ in targets:
        object_name = f"{upload_dir}/{img_file}"
        logger.debug(f"Prepared MinIO object name: {object_name}")
        # The walrus operator lets us assign the URL and test it in one expression.
        if img_url := upload_to_minio(minio_client, img_path, object_name):
            urls[img_file] = img_url

    logger.info(f"Batch image upload completed. Uploaded {len(urls)}/{len(targets)} images")
    return urls


def upload_to_minio(minio_client: Minio, local_path: str, object_name: str) -> str | None:
    """
    Upload a single local image to MinIO and return its public URL.
    """
    try:
        logger.info(f"Uploading image to MinIO. Local path={local_path}, object name={object_name}")
        minio_client.fput_object(
            bucket_name=minio_config.bucket_name,
            object_name=object_name,
            file_path=local_path,
            content_type=f"image/{os.path.splitext(local_path)[1][1:]}",
        )

        # Escape backslashes to keep the URL safe for downstream consumers.
        object_name = object_name.replace("\\", "%5C")
        protocol = "https" if minio_config.minio_secure else "http"
        base_url = f"{protocol}://{minio_config.endpoint}/{minio_config.bucket_name}"
        img_url = f"{base_url}{object_name}"
        logger.info(f"Image uploaded successfully. URL: {img_url}")
        return img_url
    except Exception as e:
        logger.error(f"Failed to upload image to MinIO: {local_path}, error: {str(e)}")
        return None


def merge_summary_and_url(summaries: Dict[str, str], urls: Dict[str, str]) -> Dict[str, Tuple[str, str]]:
    """
    Merge image summaries with upload URLs and keep only images that uploaded successfully.
    """
    image_info = {}
    for image_file, summary in summaries.items():
        if url := urls.get(image_file):
            image_info[image_file] = (summary, url)
    logger.info(f"Merged image summaries with URLs. Valid image records: {len(image_info)}")
    return image_info


def process_md_file(md_content: str, image_info: Dict[str, Tuple[str, str]]) -> str:
    """
    Replace local Markdown image references with MinIO URLs and updated alt text.
    """
    for img_filename, (summary, new_url) in image_info.items():
        pattern = re.compile(
            r"!\[.*?\]\(.*?" + re.escape(img_filename) + r".*?\)",
            re.IGNORECASE,
        )
        # Keep the simple replacement because summary and new_url are controlled plain-text values here.
        md_content = pattern.sub(f"![{summary}]({new_url})", md_content)
        logger.debug(f"Replaced Markdown image reference: {img_filename} -> {new_url}")

    logger.info(f"Markdown image replacement completed. Total replacements: {len(image_info)}")
    logger.debug(
        f"Updated Markdown content: {md_content[:500]}..."
        if len(md_content) > 500
        else f"Updated Markdown content: {md_content}"
    )
    return md_content


def step_4_upload_and_replace(
    minio_client: Minio,
    doc_stem: str,
    targets: List[Tuple[str, str, Tuple[str, str]]],
    summaries: Dict[str, str],
    md_content: str,
) -> str:
    """
    Step 4: upload images to MinIO, merge summaries and URLs, then replace local references in Markdown.
    """
    minio_img_dir = minio_config.minio_img_dir
    upload_dir = f"{minio_img_dir}/{doc_stem}".replace(" ", "")

    clean_minio_directory(minio_client, upload_dir)
    urls = upload_images_batch(minio_client, upload_dir, targets)
    image_info = merge_summary_and_url(summaries, urls)
    if image_info:
        md_content = process_md_file(md_content, image_info)

    return md_content


def step_5_backup_new_md_file(origin_md_path: str, md_content: str) -> str:
    """
    Step 5: save the processed Markdown as a new file so the original stays intact.
    """
    new_md_file_name = os.path.splitext(origin_md_path)[0] + "_new.md"
    with open(new_md_file_name, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info(f"Processed Markdown file saved successfully: {new_md_file_name}")
    return new_md_file_name


def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    Core node for Markdown image processing.

    Flow:
    1. Read Markdown content and locate the images directory.
    2. Find supported images that are actually referenced in the Markdown.
    3. Generate image summaries with the multimodal model.
    4. Upload images to MinIO and replace local references with remote URLs.
    5. Save the updated Markdown and write the new path back into state.
    """
    add_running_task(state["task_id"], sys._getframe().f_code.co_name)

    md_content, path_obj, images_dir = step_1_get_content(state)
    state["md_content"] = md_content

    if not images_dir.exists():
        logger.info(f"Images directory does not exist. Skipping image processing: {images_dir.absolute()}")
        return state

    minio_client = get_minio_client()
    if not minio_client:
        logger.warning("Failed to initialize the MinIO client. Skipping image-processing flow.")
        return state

    targets = step_2_scan_images(md_content, images_dir)
    if not targets:
        logger.info("No supported Markdown-referenced images were found. Skipping the remaining steps.")
        return state

    summaries = step_3_generate_summaries(path_obj.stem, targets)
    new_md_content = step_4_upload_and_replace(minio_client, path_obj.stem, targets, summaries, md_content)
    state["md_content"] = new_md_content

    new_md_file_name = step_5_backup_new_md_file(state["md_path"], new_md_content)
    state["md_path"] = new_md_file_name
    logger.info(f"Markdown image processing completed. New file saved to: {new_md_file_name}")

    return state


if __name__ == "__main__":
    from app.utils.path_util import PROJECT_ROOT

    logger.info(f"Local test - project root: {PROJECT_ROOT}")

    output_dir = Path(PROJECT_ROOT) / "output"
    test_md_obj = next(
        (
            path
            for path in output_dir.rglob("*.md")
            if not path.name.endswith("_new.md")
        ),
        None,
    )

    if not test_md_obj:
        logger.error(f"Local test - no Markdown test file was found under: {output_dir}")
        logger.info("Place a test Markdown file under the project's output directory and run the file again.")
    else:
        test_state = {
            "md_path": str(test_md_obj),
            "task_id": "test_task_123456",
            "md_content": "",
        }
        logger.info("Starting local test for the full Markdown image-processing flow")
        result_state = node_md_img(test_state)
        logger.info(f"Local test completed - result state: {result_state}")
