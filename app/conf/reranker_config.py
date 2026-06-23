# Core dependencies: dataclass and environment loading.
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# Load the `.env` file up front.
load_dotenv()

@dataclass
class RerankerConfig:
    bge_reranker_large: str   # Local model path
    bge_reranker_device: str  # Model device identifier
    bge_reranker_fp16: bool   # Whether FP16 is enabled (1=True/0=False)

# Instantiate the config object in the same style as lm_config.
reranker_config = RerankerConfig(
    bge_reranker_large=os.getenv("BGE_RERANKER_LARGE"),
    bge_reranker_device=os.getenv("BGE_RERANKER_DEVICE"),
    # Convert common string and numeric values from `.env` into a boolean.
    bge_reranker_fp16=os.getenv("BGE_RERANKER_FP16") in ("1", "True", "true", 1)
)
