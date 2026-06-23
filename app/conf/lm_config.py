# Core dependencies: dataclass and environment loading.
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# Load `.env` before reading environment variables so `os.getenv` can resolve them.
# If `.env` is not in the project root, you can pass an explicit path.
load_dotenv()


# LLM service configuration.
@dataclass
class LLMConfig:
    base_url: str
    api_key : str
    lv_model: str
    llm_model: str
    llm_temperature: float

lm_config = LLMConfig(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    lv_model=os.getenv("VL_MODEL"),
    llm_model=os.getenv("LLM_DEFAULT_MODEL"),
    llm_temperature=float(os.getenv("LLM_DEFAULT_TEMPERATURE"))
)
