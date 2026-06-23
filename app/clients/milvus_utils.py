import os
from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker
from app.conf.milvus_config import milvus_config
from app.core.logger import logger

# Global Milvus client instance for singleton reuse.
_milvus_client = None


def get_milvus_client():
    """
    Return the singleton Milvus client.
    Reuses the connection instead of creating a new client every time.
    :return: `MilvusClient` instance, or `None` on failure
    """
    try:
        global _milvus_client
        # Create the connection only once.
        if _milvus_client is None:
            milvus_uri = milvus_config.milvus_url
            # Validate the Milvus URI configuration.
            if not milvus_uri:
                logger.error("Failed to connect to Milvus: missing MILVUS_URL configuration")
                return None
            # Initialize the Milvus client.
            _milvus_client = MilvusClient(uri=milvus_uri)
            logger.info("Milvus client connected successfully")
        return _milvus_client
    except Exception as e:
        logger.error(f"Milvus client connection error: {str(e)}", exc_info=True)
        return None


def _coerce_int64_ids(ids):
    """
    Convert `chunk_id` values into the INT64 type expected by Milvus.
    Invalid IDs are separated from valid ones.
    :param ids: List of chunk IDs to convert
    :return: Tuple `(ok_ids, bad_ids)`
    """
    ok, bad = [], []
    for x in (ids or []):
        if x is None:
            continue
        try:
            ok.append(int(x))
        except Exception:
            bad.append(x)
    return ok, bad


def fetch_chunks_by_chunk_ids(
        client,
        collection_name: str,
        chunk_ids,
        *,
        output_fields=None,
        batch_size: int = 100,
):
    """
    Fetch chunk records from Milvus in batches by primary-key chunk ID.
    Prefer `get` for direct PK lookup, and fall back to `query` if needed.
    :param client: Milvus client instance
    :param collection_name: Target collection name
    :param chunk_ids: Chunk IDs to fetch
    :param output_fields: Fields to return
    :param batch_size: Query batch size
    :return: List of Milvus entity dictionaries
    """
    # Return early if the client or collection name is invalid.
    if client is None:
        return []
    if not collection_name:
        return []
    # Default output fields.
    if output_fields is None:
        output_fields = ["chunk_id", "content", "title", "parent_title", "item_name"]

    # Convert IDs to INT64 and separate valid ones from invalid ones.
    ok_ids, bad_ids = _coerce_int64_ids(chunk_ids)
    if bad_ids:
        logger.warning(f"Some chunk_id values could not be converted to INT64 and will be skipped: {bad_ids}")

    # Return immediately when nothing valid remains.
    if not ok_ids:
        return []

    results = []
    # Query valid IDs in batches.
    for i in range(0, len(ok_ids), batch_size):
        batch = ok_ids[i: i + batch_size]

        # Approach 1: use the primary-key `get` method when available.
        if hasattr(client, "get"):
            try:
                got = client.get(collection_name=collection_name, ids=batch, output_fields=output_fields)
                if got:
                    results.extend(got)
                continue
            except Exception as e:
                logger.warning(f"Milvus get() failed, falling back to query(): {str(e)}")

        # Approach 2: fall back to filter-based query.
        try:
            expr = f"chunk_id in [{', '.join(str(x) for x in batch)}]"
            q = client.query(collection_name=collection_name, filter=expr, output_fields=output_fields)
            if q:
                results.extend(q)
        except Exception as e:
            logger.error(f"Milvus query() batch fetch by chunk_id failed: {str(e)}", exc_info=True)

    return results


def create_hybrid_search_requests(dense_vector, sparse_vector, dense_params=None, sparse_params=None, expr=None,
                                  limit=5):
    """
    Build Milvus hybrid-search request objects for dense and sparse vectors.
    :param dense_vector: Dense vector generated from text
    :param sparse_vector: Sparse vector generated from text
    :param dense_params: Dense-vector search params, defaults to COSINE
    :param sparse_params: Sparse-vector search params, defaults to IP
    :param expr: Optional filter expression
    :param limit: Per-request result limit
    :return: List containing `[dense_req, sparse_req]`
    """
    # Default dense-vector metric.
    if dense_params is None:
        dense_params = {"metric_type": "COSINE"}
    # Default sparse-vector metric.
    if sparse_params is None:
        sparse_params = {"metric_type": "IP"}

    # Build the dense-vector ANN request.
    dense_req = AnnSearchRequest(
        data=[dense_vector],
        anns_field="dense_vector",
        param=dense_params,
        expr=expr,
        limit=limit
    )

    # Build the sparse-vector ANN request.
    sparse_req = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector",
        param=sparse_params,
        expr=expr,
        limit=limit
    )

    return [dense_req, sparse_req]


def hybrid_search(client, collection_name, reqs, ranker_weights=(0.5, 0.5), norm_score=False, limit=5,
                  output_fields=None, search_params=None):
    """
    Run a Milvus hybrid search over dense and sparse vectors.
    Uses `WeightedRanker` to combine both result sets.
    :param client: Milvus client instance
    :param collection_name: Collection name
    :param reqs: Search request list `[dense_req, sparse_req]`
    :param ranker_weights: Dense/sparse fusion weights
    :param norm_score: Whether to normalize scores before fusion
    :param limit: Final hybrid-search result limit
    :param output_fields: Fields to return, defaults to `item_name`
    :param search_params: Extra search parameters such as `ef` or `topk`
    :return: Hybrid-search result list, or `None` on failure
    """
    try:
        # Initialize the weighted ranker.
        rerank = WeightedRanker(ranker_weights[0], ranker_weights[1], norm_score=norm_score)

        # Default output fields.
        if output_fields is None:
            output_fields = ["item_name"]

        # Run the hybrid search and rerank the fused results.
        res = client.hybrid_search(
            collection_name=collection_name,
            reqs=reqs,
            ranker=rerank,
            limit=limit,
            output_fields=output_fields,
            search_params=search_params
        )

        logger.info(f"Milvus hybrid search completed, collection [{collection_name}] returned {len(res[0])} results")
        return res
    except Exception as e:
        logger.error(f"Milvus hybrid search failed for collection [{collection_name}]: {str(e)}", exc_info=True)
        return None
