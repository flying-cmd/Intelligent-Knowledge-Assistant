# Core dependencies: dataclass and environment loading.
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# Load the `.env` file up front.
load_dotenv()

# Embedding configuration for BGE-M3.
@dataclass
class EmbeddingConfig:
    bge_m3_path: str  # Local model path
    bge_m3: str       # Model repository identifier
    bge_device: str   # Runtime device (cuda:0/cpu)
    bge_fp16: bool    # Whether FP16 is enabled (1=True/0=False)

# Instantiate the config object in the same style as lm_config.
embedding_config = EmbeddingConfig(
    bge_m3_path=os.getenv("BGE_M3_PATH"),
    bge_m3=os.getenv("BGE_M3"),
    bge_device=os.getenv("BGE_DEVICE"),
    # Convert common string and numeric values from `.env` into a boolean.
    bge_fp16=os.getenv("BGE_FP16") in ("1", "True", "true", 1)
)
