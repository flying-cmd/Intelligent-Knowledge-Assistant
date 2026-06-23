from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from app.core.logger import logger
from app.conf.embedding_config import embedding_config

# Singleton model instance to avoid repeated initialization.
_bge_m3_ef = None

def get_bge_m3_ef():
    """
    Return the singleton BGE-M3 model instance using environment-based config.
    :return: Initialized `BGEM3EmbeddingFunction` instance
    """
    global _bge_m3_ef
    # Return the cached model if it has already been initialized.
    if _bge_m3_ef is not None:
        logger.debug("BGE-M3 singleton already exists, returning the cached instance")
        return _bge_m3_ef

    # Load model settings from the environment, falling back to sensible defaults.
    # Use a local path when available. Otherwise "BAAI/bge-m3" will be downloaded automatically.
    model_name = embedding_config.bge_m3_path or "BAAI/bge-m3"
    device = embedding_config.bge_device or "cpu"
    use_fp16 = embedding_config.bge_fp16 or False

    # Log the initialization parameters for troubleshooting.
    logger.info(
        "Initializing BGE-M3 model",
        extra={
            "model_name": model_name,
            "device": device,
            "use_fp16": use_fp16,
            "normalize_embeddings": True
        }
    )

    try:
        # Enable native L2 normalization so the output matches Milvus IP-based retrieval.
        _bge_m3_ef = BGEM3EmbeddingFunction(
            model_name=model_name,
            device=device,
            use_fp16=use_fp16,
            normalize_embeddings=True  # The model normalizes both dense and sparse vectors.
        )
        logger.success("BGE-M3 model initialized successfully with native L2 normalization enabled")
        return _bge_m3_ef
    except Exception as e:
        logger.error(f"Failed to initialize the BGE-M3 model: {str(e)}", exc_info=True)
        raise


def generate_embeddings(texts):
    """
    Generate dense + sparse hybrid embeddings for a list of texts.
    The model performs native L2 normalization.
    :param texts: List of input texts. Even a single text must be wrapped in a list.
    :return: Dictionary with `dense` and `sparse` embedding results
    :raise: Any embedding-generation exception, propagated to the caller
    """
    # Validate the input.
    if not isinstance(texts, list) or len(texts) == 0:
        logger.warning("Invalid embedding input: texts must be a non-empty list")
        raise ValueError("Parameter `texts` must be a non-empty list containing text")

    logger.info(f"Generating hybrid embeddings for {len(texts)} text entries")
    try:
        # Load the singleton BGE-M3 model.
        model = get_bge_m3_ef()
        # Encode the texts into dense vectors plus sparse CSR vectors.
        embeddings = model.encode_documents(texts)
        logger.debug(f"Model encoding finished, parsing sparse vectors for {len(texts)} text entries")

        # Convert sparse vectors into dictionaries for serialization and storage.
        processed_sparse = []
        for i in range(len(texts)):
            # Convert sparse indices from np.int64 to Python int for safe dict keys.
            sparse_indices = embeddings["sparse"].indices[
                embeddings["sparse"].indptr[i]:embeddings["sparse"].indptr[i + 1]
            ].tolist()
            # Convert sparse weights from np.float32 to Python float for JSON compatibility.
            sparse_data = embeddings["sparse"].data[
                embeddings["sparse"].indptr[i]:embeddings["sparse"].indptr[i + 1]
            ].tolist()
            # Build a sparse vector dictionary in the form {feature_index: normalized_weight}.
            sparse_dict = {k: v for k, v in zip(sparse_indices, sparse_data)}
            processed_sparse.append(sparse_dict)

        # Build the final return value and convert NumPy arrays to plain lists.
        result = {
            "dense": [emb.tolist() for emb in embeddings["dense"]],  # One nested list per input text
            "sparse": processed_sparse  # List of dicts, already L2-normalized by the model
        }
        logger.success(f"Generated embeddings for {len(texts)} text entries in production-ready format")
        return result

    except Exception as e:
        logger.error(f"Failed to generate text embeddings: {str(e)}", exc_info=True)
        raise
