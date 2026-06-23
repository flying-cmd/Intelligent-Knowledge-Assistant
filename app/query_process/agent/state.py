from typing_extensions import TypedDict
from typing import List


class QueryGraphState(TypedDict):
    """
    State object shared across the query workflow.
    """
    session_id: str  # Unique session identifier
    original_query: str  # Original user question

    # Intermediate retrieval data
    embedding_chunks: list  # Chunks returned by standard embedding search
    hyde_embedding_chunks: list  # Chunks returned by HyDE retrieval
    kg_chunks: list  # Chunks returned by knowledge-graph retrieval
    web_search_docs: list  # Documents returned by web search

    # Ranking-stage data
    rrf_chunks: list  # Chunks after RRF fusion
    reranked_docs: list  # Final Top-K documents after reranking

    # Generation-stage data
    prompt: str  # Assembled prompt
    answer: str  # Final generated answer

    # Supporting metadata
    item_names: List[str]  # Extracted product names
    rewritten_query: str  # Rewritten query
    history: list  # Conversation history
    is_stream: bool  # Whether streaming output is enabled
