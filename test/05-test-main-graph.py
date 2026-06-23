
from app.core.logger import logger
from app.import_process.agent.main_graph import kb_import_app
from app.import_process.agent.state import ImportGraphState

if __name__ == "__main__":
    from app.utils.path_util import PROJECT_ROOT
    import os
    from pathlib import Path

    # End-to-end test: verify the full PDF import -> Milvus -> KG pipeline.
    logger.info("===== Starting end-to-end knowledge graph import test =====")
    # 1. Build the test file path by selecting the first PDF in the doc folder.
    doc_dir = Path(PROJECT_ROOT) / "doc"
    test_pdf_path_obj = next(doc_dir.glob("*.pdf"), None)
    test_pdf_path = str(test_pdf_path_obj) if test_pdf_path_obj else ""
    # 2. Build the output directory for intermediate files.
    test_output_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(test_output_dir, exist_ok=True)  # Create it if it does not exist.

    # 3. Validate that the test PDF exists.
    if not test_pdf_path or not os.path.exists(test_pdf_path):
        logger.error(f"End-to-end test failed: test PDF does not exist: {test_pdf_path}")
        logger.info("Check the file path or place the test file in the project's doc folder.")
    else:
        # 4. Build the test state.
        test_state = ImportGraphState({
            "task_id": "test_kg_import_workflow_001",  # Test task ID
            "user_id": "test_user",  # Test user ID
            "local_file_path": test_pdf_path,  # Test PDF path
            "local_dir": test_output_dir,  # Intermediate output directory
            "is_pdf_read_enabled": False,  # Enable PDF parsing
            "is_md_read_enabled": False  # Disable Markdown parsing
        })
        try:
            logger.info(f"Test task started, PDF path: {test_pdf_path}")
            logger.info(f"Intermediate output directory: {test_output_dir}")
            logger.info("Running the full node pipeline: entry -> pdf2md -> md_img -> split -> item_name -> embedding -> milvus -> kg")

            # 5. Execute the full LangGraph pipeline.
            final_state = None
            for step in kb_import_app.stream(test_state, stream_mode="values"):
                # Print the node that just finished.
                current_node = list(step.keys())[-1] if step else "unknown_node"
                logger.info(f"Node finished: {current_node}")
                final_state = step  # Save the final state.

            # 6. Print a preview of the results and core metrics.
            if final_state:
                logger.info("-" * 80)
                logger.info("===== End-to-end test completed successfully: core result preview =====")
                # Extract key metrics.
                chunks = final_state.get("chunks", [])
                chunk_count = len(chunks)
                md_content = final_state.get("md_content", "")[:150]  # First 150 characters of Markdown content
                has_embedding = all("dense_vector" in c and "sparse_vector" in c for c in chunks) if chunks else False
                has_chunk_id = all("chunk_id" in c for c in chunks) if chunks else False
                kg_id = final_state.get("kg_id", "not_generated")  # KG import ID

                # Print key metrics.
                logger.info(f"PDF-to-Markdown preview (first 150 characters): {md_content}...")
                logger.info(f"Total document chunk count: {chunk_count}")
                logger.info(f"Did every chunk finish vectorization: {'yes' if has_embedding else 'no'}")
                logger.info(f"Did every chunk finish Milvus import (including chunk_id): {'yes' if has_chunk_id else 'no'}")
                logger.info(f"Knowledge graph import ID: {kg_id}")
                logger.info(f"Core keys present in the final state: {list(final_state.keys())}")
                logger.info("-" * 80)
        except Exception as e:
            # 7. Catch exceptions and print the detailed error.
            logger.error("===== End-to-end test failed =====", exc_info=True)
            logger.error(f"Error reason: {str(e)}")
    logger.info("===== End-to-end knowledge graph import test finished =====")
