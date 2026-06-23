import re
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.utils.task_utils import add_running_task
from app.import_process.agent.state import ImportGraphState
from app.core.logger import logger

# --- Configuration ---
# Maximum characters allowed in a single chunk before a second split pass is triggered.
DEFAULT_MAX_CONTENT_LENGTH = 2000
# Neighboring short chunks under the same parent title can be merged to reduce fragmentation.
MIN_CONTENT_LENGTH = 500


def step_1_get_inputs(state: ImportGraphState) -> Tuple[Any, str, int]:
    """
    Step 1: read and normalize the input Markdown data.
    """
    content = state.get("md_content")
    if not content:
        logger.warning("No valid Markdown content was found in workflow state. Document splitting will stop.")
        return None, None, None

    # Normalize line endings so downstream processing behaves consistently across platforms.
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    file_title = state.get("file_title", "Unknown File")
    max_len = DEFAULT_MAX_CONTENT_LENGTH

    logger.info(f"Step 1: input data loaded successfully. File title: {file_title}, max chunk length: {max_len}")
    return content, file_title, max_len


def step_2_split_by_titles(content: str, file_title: str) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Step 2: split the Markdown document by headings while ignoring heading-like text inside code blocks.
    """
    title_pattern = r"^\s*#{1,6}\s+.+"

    lines = content.split("\n")
    sections = []
    current_title = ""
    current_lines = []
    title_count = 0
    in_code_block = False

    def _flush_section():
        """Write the current buffered section into the results list."""
        if not current_lines:
            return
        sections.append(
            {
                "title": current_title,
                "content": "\n".join(current_lines),
                "file_title": file_title,
            }
        )

    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith("```") or stripped_line.startswith("~~~"):
            in_code_block = not in_code_block
            current_lines.append(line)
            continue

        is_valid_title = (not in_code_block) and re.match(title_pattern, line)
        if is_valid_title:
            _flush_section()
            current_title = line.strip()
            current_lines = [current_title]
            title_count += 1
            logger.debug(f"Detected Markdown heading: {current_title}")
        else:
            current_lines.append(line)

    _flush_section()
    logger.info(f"Step 2: heading split completed. Valid headings: {title_count}, total source lines: {len(lines)}")
    return sections, title_count, len(lines)


def step_3_handle_no_title(
    content: str,
    sections: List[Dict[str, Any]],
    title_count: int,
    file_title: str,
) -> List[Dict[str, Any]]:
    """
    Step 3: provide a fallback when no headings were detected.
    """
    if title_count == 0:
        logger.warning(f"Step 3: no Markdown headings were detected. Processing the full file as one section: {file_title}")
        return [{"title": "Untitled", "content": content, "file_title": file_title}]

    logger.debug(f"Step 3: detected {title_count} valid headings, so no fallback handling is needed")
    return sections


def _split_long_section(
    section: Dict[str, Any],
    max_length: int = DEFAULT_MAX_CONTENT_LENGTH,
) -> List[Dict[str, Any]]:
    """
    Helper: split an oversized section from coarse units down to fine-grained units while preserving context.
    """
    content = section.get("content", "") or ""
    if len(content) <= max_length:
        return [section]

    content = content.replace("\r\n", "\n").replace("\r", "\n")
    title = section.get("title", "") or ""
    prefix = f"{title}\n\n" if title else ""
    available_len = max_length - len(prefix)
    if available_len <= 0:
        logger.warning(f"Section title is too long to split safely: {title[:20]}...")
        return [section]

    body = content
    if title and body.lstrip().startswith(title):
        body = body[body.find(title) + len(title):].lstrip()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=available_len,
        chunk_overlap=0,
        # Use Unicode escapes for full-width sentence punctuation so Chinese documents still split cleanly.
        separators=["\n\n", "\n", "\u3002", "\uFF01", "\uFF1F", "\uFF1B", ".", "!", "?", ";", " "],
    )

    sub_sections = []
    for idx, chunk in enumerate(splitter.split_text(body), start=1):
        text = chunk.strip()
        if not text:
            continue

        full_text = (prefix + text).strip()
        sub_sections.append(
            {
                "title": f"{title}-{idx}" if title else f"chunk-{idx}",
                "content": full_text,
                "parent_title": title,
                "part": idx,
                "file_title": section.get("file_title"),
            }
        )

    logger.debug(f"Long-section split completed: {title} -> generated {len(sub_sections)} sub-chunks")
    return sub_sections


def _merge_short_sections(
    sections: List[Dict[str, Any]],
    min_length: int = MIN_CONTENT_LENGTH,
) -> List[Dict[str, Any]]:
    """
    Helper: merge short neighboring chunks that belong to the same parent section.
    """
    if not sections:
        logger.debug("The chunk list to merge is empty; returning immediately")
        return []

    merged_sections = []
    current_chunk = None

    for sec in sections:
        if current_chunk is None:
            current_chunk = sec
            continue

        is_current_short = len(current_chunk["content"]) < min_length
        is_same_parent = current_chunk.get("parent_title") == sec.get("parent_title")

        if is_current_short and is_same_parent:
            parent_title = sec.get("parent_title", "")
            next_content = sec["content"]
            if parent_title and next_content.startswith(parent_title):
                next_content = next_content[len(parent_title):].lstrip()

            current_chunk["content"] += "\n\n" + next_content
            if "part" in sec:
                current_chunk["part"] = sec["part"]
            logger.debug(
                f"Merged short chunk under parent title {current_chunk.get('parent_title')}, "
                f"accumulated length: {len(current_chunk['content'])}"
            )
        else:
            merged_sections.append(current_chunk)
            current_chunk = sec

    if current_chunk is not None:
        merged_sections.append(current_chunk)

    logger.debug(f"Short-chunk merge completed: {len(sections)} -> {len(merged_sections)}")
    return merged_sections


def step_4_refine_chunks(sections: List[Dict[str, Any]], max_len: int) -> List[Dict[str, Any]]:
    """
    Step 4: refine chunk sizes by splitting long sections, merging short ones, and filling required metadata.
    """
    if not max_len or max_len <= 0:
        logger.warning(f"Step 4: invalid max chunk length configuration ({max_len}); skipping refinement")
        return sections

    refined_split = []
    for sec in sections:
        # extend() appends each generated sub-chunk directly into the target list.
        refined_split.extend(_split_long_section(sec, max_len))
    logger.info(f"Step 4-1: long-section split completed. Generated {len(refined_split)} initial sub-chunks")

    final_sections = _merge_short_sections(refined_split)
    logger.info(f"Step 4-2: short-section merge completed. Final chunk count: {len(final_sections)}")

    for sec in final_sections:
        if not isinstance(sec, dict):
            continue

        if "part" not in sec:
            sec["part"] = 0

        if not sec.get("parent_title"):
            sec["parent_title"] = sec.get("title") or ""

    logger.debug("Step 4-3: parent_title fallback completed for all chunks")
    return final_sections


def step_5_print_stats(lines_count: int, sections: List[Dict[str, Any]]) -> None:
    """
    Step 5: print document-splitting statistics for monitoring and debugging.
    """
    chunk_num = len(sections)
    logger.info("-" * 50 + " Document Split Statistics " + "-" * 50)
    logger.info(f"Original Markdown line count: {lines_count}")
    logger.info(f"Final generated chunk count: {chunk_num}")
    if sections:
        first_title = sections[0].get("title", "Untitled")
        logger.info(f"First chunk title preview: {first_title}")
    logger.info("-" * 110)


def step_6_backup(state: ImportGraphState, sections: List[Dict[str, Any]]) -> None:
    """
    Step 6: back up the final chunk list as local JSON for debugging and issue investigation.
    """
    local_dir = state.get("local_dir")
    if not local_dir:
        logger.warning("Step 6: backup directory (local_dir) is not configured; skipping chunk backup")
        return

    try:
        os.makedirs(local_dir, exist_ok=True)
        backup_path = os.path.join(local_dir, "chunks.json")
        with open(backup_path, "w", encoding="utf-8") as f:
            # json.dump serializes native Python lists/dicts directly into JSON, which is exactly what we want here.
            json.dump(
                sections,
                f,
                ensure_ascii=False,  # Preserve non-ASCII text instead of escaping it.
                indent=2,  # Keep the backup human-readable.
            )
        logger.info(f"Step 6: chunk backup completed successfully: {backup_path}")
    except Exception as e:
        logger.error(f"Step 6: failed to back up chunk results: {str(e)}", exc_info=False)


def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    Core node: split a Markdown document into retrieval-friendly chunks.

    Flow:
    - Load input
    - Split by headings
    - Apply no-heading fallback
    - Refine chunk sizes
    - Print statistics
    - Back up the results
    """
    node_name = sys._getframe().f_code.co_name
    logger.info(f">>> Starting core node: [Document Split] {node_name}")
    add_running_task(state["task_id"], node_name)

    try:
        content, file_title, max_len = step_1_get_inputs(state)
        if content is None:
            logger.info(f">>> Node execution stopped: {node_name} (no valid Markdown content)")
            return state

        sections, title_count, lines_count = step_2_split_by_titles(content, file_title)
        sections = step_3_handle_no_title(content, sections, title_count, file_title)
        sections = step_4_refine_chunks(sections, max_len)
        step_5_print_stats(lines_count, sections)

        state["chunks"] = sections
        step_6_backup(state, sections)

        logger.info(
            f">>> Core node completed: [Document Split] {node_name}, "
            f"generated {len(sections)} valid chunks and wrote them back to workflow state"
        )

    except Exception as e:
        logger.error(f">>> Core node failed: [Document Split] {node_name}, error: {str(e)}", exc_info=True)

    return state


if __name__ == "__main__":
    from app.utils.path_util import PROJECT_ROOT
    from app.import_process.agent.nodes.node_md_img import node_md_img

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
        logger.info("Place a test Markdown file in the project's output directory and run the file again.")
    else:
        test_state = {
            "md_path": str(test_md_obj),
            "task_id": "test_task_123456",
            "md_content": "",
            "file_title": test_md_obj.stem,
            "local_dir": os.path.join(PROJECT_ROOT, "output"),
        }
        logger.info("Starting local test for the full Markdown image-processing flow")
        result_state = node_md_img(test_state)
        logger.info(f"Local test completed - preprocessing state: {result_state}")
        logger.info("\n=== Starting integration test for node_document_split ===")

        logger.info(">> Running node_document_split")
        final_state = node_document_split(result_state)
        final_chunks = final_state.get("chunks", [])
        logger.info(f"Test succeeded: generated {len(final_chunks)} valid chunks: {final_chunks}")
