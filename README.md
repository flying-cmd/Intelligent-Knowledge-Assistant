# Intelligent Knowledge Assistant

Intelligent Knowledge Assistant is a FastAPI + LangGraph knowledge-base assistant for product manuals and technical documents.

It supports two major workflows:

1. Import documents into a retrieval-ready knowledge base.
2. Answer user questions with a multi-route RAG pipeline that combines local vector search, HyDE retrieval, reranking, chat history, and optional web search.

The repository is currently geared toward manuals in PDF/Markdown form and includes sample documents under [`doc/`](./doc).

## What The Project Does

This project helps turn product manuals into a searchable assistant:

- Upload a `.pdf` or `.md` file.
- Convert PDFs to Markdown through MinerU.
- Process document images and replace local image references with MinIO URLs.
- Split Markdown into retrieval chunks.
- Recognize the main product/item name from the document.
- Generate hybrid dense + sparse embeddings with BGE-M3.
- Store chunk vectors in Milvus.
- Query the knowledge base through a chat API.
- Merge local retrieval results with optional web search results.
- Rerank the evidence and generate a final answer with an OpenAI-compatible LLM API.

## Main Components

### Import service

The import service lives in [`app/import_process/api/file_import_service.py`](./app/import_process/api/file_import_service.py) and exposes:

- `GET /import.html` for the built-in upload page
- `POST /upload` for file upload and background processing
- `GET /status/{task_id}` for import progress tracking

The import workflow is compiled in [`app/import_process/agent/main_graph.py`](./app/import_process/agent/main_graph.py).

### Query service

The query service lives in [`app/query_process/api/query_service.py`](./app/query_process/api/query_service.py) and exposes:

- `GET /chat.html` for the built-in chat page
- `GET /health` for health checks
- `POST /query` for sync or async question answering
- `GET /stream/{session_id}` for SSE streaming
- `GET /history/{session_id}` for chat history
- `DELETE /history/{session_id}` to clear chat history

The query workflow is compiled in [`app/query_process/agent/main_graph.py`](./app/query_process/agent/main_graph.py).

## Architecture

### Import flow

```text
Upload file
  -> node_entry
  -> node_pdf_to_md      (PDF only, via MinerU)
  -> node_md_img         (image summary + MinIO upload + Markdown rewrite)
  -> node_document_split
  -> node_item_name_recognition
  -> node_bge_embedding
  -> node_import_milvus
  -> done
```

### Query flow

```text
User question
  -> node_item_name_confirm
  -> parallel retrieval
       - node_search_embedding
       - node_search_embedding_hyde
       - node_query_kg
       - node_web_search_mcp
  -> node_rrf
  -> node_rerank
  -> node_answer_output
  -> done
```

### Storage and external systems

- `Milvus`: vector store for chunk retrieval and item-name lookup
- `MinIO`: object storage for uploaded PDFs and processed images
- `MongoDB`: conversation history storage
- `MinerU`: PDF-to-Markdown parsing service
- `OpenAI-compatible LLM endpoint`: answer generation, item-name recognition, image summarization
- `Bailian MCP web search`: optional search branch during query flow
- `Neo4j`: intended knowledge-graph integration, though the current query node is only a stub

## Repository Layout

```text
app/
  clients/         External service clients for Milvus, MinIO, MongoDB, Neo4j
  conf/            Environment-backed config objects
  core/            Shared logger and prompt loader
  import_process/  Import API, LangGraph state, nodes, and upload page
  lm/              LLM, embedding, and reranker helpers
  query_process/   Query API, LangGraph state, nodes, and chat page
  tool/            Model download helper scripts
  utils/           SSE, path, task, and formatting utilities
doc/               Sample source documents
prompts/           Prompt templates used by the workflows
test/              Script-style smoke tests and experiments
docker-compose.yml Milvus + MinIO + etcd local stack
pyproject.toml     Python dependencies
```

## Technology Stack

- Python 3.11+
- FastAPI
- Uvicorn
- LangGraph
- LangChain
- Milvus
- MinIO
- MongoDB
- Loguru
- FlagEmbedding / BGE models
- OpenAI-compatible chat API

## Prerequisites

Before running the project, make sure you have:

- Python `3.11` or newer
- `uv` installed, or `pip` if you prefer
- Docker Desktop or Docker Engine for the local Milvus stack
- Access to the following external services or equivalents:
  - MinerU
  - MongoDB
  - OpenAI-compatible LLM API

## Quick Start

### 1. Clone and enter the project

```powershell
git clone <your-repo-url>
cd RAG_copy
```

### 2. Create the environment

Using `uv`:

```powershell
uv sync
```

Using `pip`:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

### 3. Create `.env`

Create a project-root `.env` file based on the template below.

```env

#MINERU
MINERU_MODEL_SOURCE=modelscope
MODELSCOPE_OFFLINE=1
MODELSCOPE_CACHE=modelscope cache path
HF_HOME=huggingface cache path
MD_ROOT_DIR=./temp-files/


# API key request page: https://bailian.console.aliyun.com/cn-beijing/?spm=5176.29597918.J_SEsSjsNv72yRuRFS2VknO.2.4d877b08ThdGtP&tab=model#/api-key
# Large model docs: https://bailian.console.aliyun.com/cn-beijing/?spm=5176.29597918.J_SEsSjsNv72yRuRFS2VknO.2.4d877b08ThdGtP&tab=doc#/doc
LLM_DEFAULT_MODEL=qwen-flash
VL_MODEL=qwen3-vl-flash
OPENAI_API_KEY=xxxxx
OPENAI_BASE_URL=xxxx
LLM_DEFAULT_TEMPERATURE=0.1


# Embedding model used for vector generation, downloaded locally through ModelScope.
BGE_M3_PATH=model path
BGE_M3=BAAI/bge-m3
# gpu
# BGE_DEVICE=cuda:0
# cpu
BGE_DEVICE=cpu
# Set this to 0 on CPU. GPU vector generation usually benefits from FP16, while CPU typically uses FP32.
BGE_FP16=0
ITEM_NAME_DIAG=1

# Milvus configuration
# Replace this with your Milvus URL.
MILVUS_URL=http://localhost:19530
# Collection used to store chunks
CHUNKS_COLLECTION=kb_chunks
# Collection used to store the entity type for each document
ITEM_NAME_COLLECTION=kb_item_names

EMBEDDING_DIM=1024

# MongoDB configuration
MONGO_URL=http://localhost:27017
MONGO_DB_NAME=kb002

# MinIO client
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET_NAME=knowledge-base-files
MINIO_IMG_DIR=/upload-images
MINIO_SECURE=False
# Reranker model configuration
BGE_RERANKER_LARGE=model path
#BGE_RERANKER_DEVICE=cuda:0
BGE_RERANKER_DEVICE=cpu
#BGE_RERANKER_FP16=1
BGE_RERANKER_FP16=0


MCP_DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/sse


# MinerU API token and base URL
MINERU_API_TOKEN=xxxxxx
MINERU_BASE_URL=https://mineru.net/api/v4



# ===================== Logging Configuration =====================
# Console logging: True = enabled / False = disabled. Level options: DEBUG/INFO/WARNING/ERROR/CRITICAL
LOG_CONSOLE_ENABLE=True
LOG_CONSOLE_LEVEL=INFO

# File logging
# True = enabled / False = disabled. The level can be configured independently from the console logger.
LOG_FILE_ENABLE=True
LOG_FILE_LEVEL=INFO
# Log retention period (expired logs are deleted automatically to avoid filling the disk)
LOG_FILE_RETENTION=7 days
```

## Start The Infrastructure

The included Docker Compose file starts:

- `etcd`
- `minio`
- `milvus standalone`

Run:

```powershell
docker compose up -d
```

Useful local endpoints from `docker-compose.yml`:

- MinIO API: `http://127.0.0.1:9000`
- MinIO console: `http://127.0.0.1:9001`
- Milvus: `http://127.0.0.1:19530`
- Milvus health: `http://127.0.0.1:9091/healthz`

Note: MongoDB, MinerU, and the LLM endpoint are not started by this compose file. You must provide them separately.

## Run The Services

### Import API

```powershell
uv run python app/import_process/api/file_import_service.py
```

Service URL:

- `http://127.0.0.1:8000`
- Built-in page: `http://127.0.0.1:8000/import.html`

### Query API

```powershell
uv run python app/query_process/api/query_service.py
```

Service URL:

- `http://127.0.0.1:8001`
- Built-in page: `http://127.0.0.1:8001/chat.html`

## API Usage

### Import a file

```bash
curl -X POST "http://127.0.0.1:8000/upload" \
  -F "files=@doc/sample.pdf"
```

Example response:

```json
{
  "code": 200,
  "message": "Files uploaded successfully, total: 1",
  "task_ids": ["<task-id>"]
}
```

### Check import progress

```bash
curl "http://127.0.0.1:8000/status/<task-id>"
```

### Sync query

```bash
curl -X POST "http://127.0.0.1:8001/query" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"How do I configure the device?\",\"is_stream\":false}"
```

### Async query with SSE

Start the task:

```bash
curl -X POST "http://127.0.0.1:8001/query" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"How do I configure the device?\",\"is_stream\":true}"
```

Then listen for events:

```bash
curl -N "http://127.0.0.1:8001/stream/<session-id>"
```

### Read conversation history

```bash
curl "http://127.0.0.1:8001/history/<session-id>"
```

### Clear conversation history

```bash
curl -X DELETE "http://127.0.0.1:8001/history/<session-id>"
```

## Prompt Files

Prompt templates are stored in [`prompts/`](./prompts) and loaded by [`app/core/load_prompt.py`](./app/core/load_prompt.py).

Current prompt files include:

- `answer_out.prompt`
- `hyde_prompt.prompt`
- `image_summary.prompt`
- `item_name_recognition.prompt`
- `product_recognition_system.prompt`
- `rewritten_query_and_itemnames.prompt`

These prompts drive:

- answer generation
- query rewriting
- product-name extraction
- image summarization
- HyDE retrieval

## Model Helpers

The repository includes helper scripts for local model download:

- [`app/tool/download_bgem3.py`](./app/tool/download_bgem3.py)
- [`app/tool/download_reranker.py`](./app/tool/download_reranker.py)

These are convenience scripts only. Their current paths are hard-coded, so adjust them before use if your local model directory differs.

## Testing

The `test/` directory currently contains script-style tests rather than a unified `pytest` suite.

Useful files:

- [`test/01-env-and-system-env-priority.py`](./test/01-env-and-system-env-priority.py)
- [`test/02-logger-test.py`](./test/02-logger-test.py)
- [`test/03-cuda-test.py`](./test/03-cuda-test.py)
- [`test/04-test_graph_flow.py`](./test/04-test_graph_flow.py)
- [`test/05-test-main-graph.py`](./test/05-test-main-graph.py)

Example:

```powershell
uv run python test/04-test_graph_flow.py
uv run python test/05-test-main-graph.py
```

## Runtime Data And Generated Files

The project writes runtime artifacts to:

- `logs/` for Loguru output
- `output/` for uploaded files and intermediate processing results

The import API stores uploaded files under:

```text
output/YYYYMMDD/<task-id>/
```

## Troubleshooting

### The app cannot find the project root

Make sure either:

- the project root contains `.env`, or
- `PROJECT_ROOT` is set explicitly

This is resolved by [`app/utils/path_util.py`](./app/utils/path_util.py).

### Milvus connection fails

Check:

- `MILVUS_URL`
- whether `docker compose up -d` completed successfully
- whether port `19530` is reachable

### MinIO upload or image URLs fail

Check:

- `MINIO_ENDPOINT`
- `MINIO_ACCESS_KEY`
- `MINIO_SECRET_KEY`
- `MINIO_BUCKET_NAME`
- whether MinIO is reachable on port `9000`

### LLM calls fail immediately

Check:

- `OPENAI_BASE_URL`
- `OPENAI_API_KEY`
- `LLM_DEFAULT_MODEL`
- `VL_MODEL`

### Query history fails

Check:

- `MONGO_URL`
- `MONGO_DB_NAME`
- whether MongoDB is running and reachable

## Suggested First End-To-End Run

1. Start Docker Compose for Milvus + MinIO.
2. Prepare `.env` with valid LLM, MinerU, and MongoDB settings.
3. Start the import API on port `8000`.
4. Upload one sample PDF from `doc/`.
5. Poll the task status until it completes.
6. Start the query API on port `8001`.
7. Open `/chat.html` or call `/query` directly.

