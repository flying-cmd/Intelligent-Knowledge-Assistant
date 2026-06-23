# Environment configuration and dependencies.
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.exceptions import LangChainException
from typing import Optional

# Project-internal dependencies.
from app.conf.lm_config import lm_config
from app.core.logger import logger

# Global cache: key is `(model_name, json_mode)`, value is a `ChatOpenAI` instance.
# This avoids repeated initialization and improves performance.
_llm_client_cache = {}


def get_llm_client(model: Optional[str] = None, json_mode: bool = False) -> ChatOpenAI:
    """
    Return a LangChain `ChatOpenAI` client instance backed by a global cache.
    Supports OpenAI-compatible APIs such as OpenAI and Qwen, plus optional
    JSON output mode.

    :param model: Model name. Priority: function argument > `lm_config.llm_model` > `qwen3-32b`
    :param json_mode: Whether to force JSON-object output
    :return: Initialized `ChatOpenAI` instance
    :raise ValueError: Missing API key, base URL, or other required config
    :raise Exception: Model initialization failure from the LangChain layer
    """
    # 1. Determine the target model.
    target_model = model or lm_config.llm_model or "qwen3-32b"
    # Cache key: model name + JSON mode.
    cache_key = (target_model, json_mode)

    # 2. Return the cached instance when available.
    if cache_key in _llm_client_cache:
        logger.debug(f"[LLM client] Cache hit, returning existing instance: model={target_model}, json_mode={json_mode}")
        return _llm_client_cache[cache_key]

    # 3. Validate required API configuration.
    if not lm_config.api_key:
        raise ValueError("[LLM client] Missing configuration: set OPENAI_API_KEY in .env")
    if not lm_config.base_url:
        raise ValueError("[LLM client] Missing configuration: set OPENAI_API_BASE or OPENAI_BASE_URL in .env")
    logger.info(f"[LLM client] Initializing new instance: model={target_model}, json_mode={json_mode}")

    # 4. Build request parameters.
    # `extra_body` contains provider-specific options transparently forwarded by LangChain.
    extra_body = {"enable_thinking": False}  # Qwen-specific: disable thinking-chain output.
    # `model_kwargs` contains standard OpenAI-compatible parameters.
    model_kwargs = {}
    if json_mode:
        # Force the model to return a parseable JSON object.
        model_kwargs["response_format"] = {"type": "json_object"}
        logger.debug("[LLM client] JSON output mode enabled")

    # 5. Initialize the client and wrap LangChain errors with clearer messages.
    try:
        llm_client = ChatOpenAI(
            model=target_model,
            temperature=lm_config.llm_temperature or 0.1,
            api_key=lm_config.api_key,
            base_url=lm_config.base_url,
            extra_body=extra_body,
            model_kwargs=model_kwargs,
        )
    except LangChainException as e:
        raise Exception(f"[LLM client] Failed to initialize model `{target_model}` at the LangChain layer: {str(e)}") from e

    # 6. Cache the new instance for reuse.
    _llm_client_cache[cache_key] = llm_client
    logger.info(f"[LLM client] Instance initialized and cached: model={target_model}, json_mode={json_mode}")

    return llm_client


# Local test: verify client creation, cache behavior, and logging.
if __name__ == "__main__":
    logger.info("===== Starting LLM client utility test =====")
    try:
        # Test 1: default configuration.
        client1 = get_llm_client()
        logger.info("Test 1 passed: default client created successfully")

        # Test 2: specific multimodal model in normal mode.
        client2 = get_llm_client(model="qwen-vl-plus")
        logger.info("Test 2 passed: client for a specific multimodal model created successfully")

        # Test 3: same model and mode, verify cache reuse.
        client3 = get_llm_client(model="qwen-vl-plus")
        logger.info(f"Test 3 passed: cache reuse verified, client2 is client3: {client2 is client3}")

        # Test 4: JSON output mode.
        client4 = get_llm_client(model="qwen3-32b", json_mode=True)
        logger.info("Test 4 passed: JSON output mode client created successfully")

    except Exception as e:
        logger.error(f"LLM client utility test failed: {str(e)}", exc_info=True)
    finally:
        logger.info("===== LLM client utility test finished =====")
