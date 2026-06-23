import json

from app.import_process.agent.main_graph import kb_import_app
from app.import_process.agent.state import create_default_state
import sys
from app.core.logger import logger

logger.info("===== Test started =====")

initial_state = create_default_state(local_file_path="Using-the-RS-12-multimeter.pdf")
final_state = None

# Only print the final state value (dictionary form), without extra node names,
# execution logs, or metadata.
for event in kb_import_app.stream(initial_state):
    for key, value in event.items():
        logger.info(f"Node: {key}")
        final_state = value

# Print the final state in formatted JSON.
logger.info(f"Final state: \n {json.dumps(final_state, indent=4, ensure_ascii=False)}")

logger.info("Graph structure:")
# uv add grandalf
kb_import_app.get_graph().print_ascii()

logger.info("===== Test finished =====")
