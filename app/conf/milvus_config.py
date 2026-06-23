# Core dependencies shared by the configuration classes.
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# Load `.env` once for the whole process.
load_dotenv()

# ===================== Other config classes can live above this block =====================
# ... your LLMConfig and EmbeddingConfig code ...

# Milvus vector database configuration.
@dataclass
class MilvusConfig:
    milvus_url: str          # Milvus server URL
    chunks_collection: str   # Collection used to store chunks
    entity_name_collection: str  # Reserved entity-name collection
    item_name_collection: str    # Collection used to store document item names

# Instantiate the Milvus config object.
milvus_config = MilvusConfig(
    milvus_url=os.getenv("MILVUS_URL"),
    chunks_collection=os.getenv("CHUNKS_COLLECTION"),
    entity_name_collection=os.getenv("ENTITY_NAME_COLLECTION"),
    item_name_collection=os.getenv("ITEM_NAME_COLLECTION")
)
