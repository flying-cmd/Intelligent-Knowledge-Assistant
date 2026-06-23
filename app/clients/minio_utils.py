# Python standard library imports.
import os
import json
# Core MinIO SDK class.
from minio import Minio
# Project configuration and logging.
from app.conf.minio_config import minio_config
from app.core.logger import logger

# Global MinIO client initialized once for project-wide reuse.
minio_client = None

try:
    # Initialize the MinIO client instance.
    minio_client = Minio(
        endpoint=minio_config.endpoint,
        access_key=minio_config.access_key,
        secret_key=minio_config.secret_key,
        secure=False  # Use HTTP for local/private deployments. Use HTTPS with SSL for public deployments.
    )
    bucket_name = minio_config.bucket_name

    # Create the bucket if it does not already exist.
    if not minio_client.bucket_exists(bucket_name):
        logger.info(f"MinIO bucket [{bucket_name}] does not exist, creating it now")
        minio_client.make_bucket(bucket_name)
        logger.info(f"MinIO bucket [{bucket_name}] created successfully")
    else:
        logger.info(f"MinIO bucket [{bucket_name}] already exists, skipping creation")

    # Configure a public read-only bucket policy so files can be accessed by URL.
    bucket_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"AWS": ["*"]},  # "*" means anonymous users in S3-compatible policy syntax.
            "Action": ["s3:GetObject"],   # Allow object-read access only.
            "Resource": [f"arn:aws:s3:::{bucket_name}/*"]
        }]
    }
    minio_client.set_bucket_policy(bucket_name, json.dumps(bucket_policy))
    logger.info(f"MinIO bucket [{bucket_name}] configured with public read-only access")

except Exception as e:
    # Reset the client to `None` if initialization fails.
    logger.error(f"Failed to initialize the MinIO client: {str(e)}", exc_info=True)
    minio_client = None


def get_minio_client():
    """
    Return the globally initialized MinIO client.
    :return: MinIO client instance, or `None` if initialization failed
    """
    return minio_client
