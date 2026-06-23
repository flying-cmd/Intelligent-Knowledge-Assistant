# Core dependencies: dataclass and environment loading.
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


# MCP service configuration.
@dataclass
class McpConfig:
    mcp_base_url: str
    api_key : str

mcp_config = McpConfig(
    mcp_base_url=os.getenv("MCP_DASHSCOPE_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)
