# Core dependencies: dataclass and environment loading.
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# Load `.env` up front so MinIO settings are available via `os.getenv`.
load_dotenv()


# MinIO object storage configuration.
@dataclass
class MinIOConfig:
    endpoint: str       # MinIO endpoint including scheme and port
    access_key: str     # MinIO access key
    secret_key: str     # MinIO secret key
    bucket_name: str    # Default MinIO bucket for knowledge-base files
    minio_img_dir: str  # Folder used to store images in MinIO
    minio_secure: bool  # Whether to use SSL / HTTPS


# Instantiate the MinIO config from `.env`.
minio_config = MinIOConfig(
    endpoint=os.getenv("MINIO_ENDPOINT"),
    access_key=os.getenv("MINIO_ACCESS_KEY"),
    secret_key=os.getenv("MINIO_SECRET_KEY"),
    bucket_name=os.getenv("MINIO_BUCKET_NAME"),
    minio_img_dir=os.getenv("MINIO_IMG_DIR"),
    minio_secure=os.getenv("MINIO_SECURE") == "True"
)
