## geniusai-server

The python backend for LrGeniusAI and possibly others in the future...

[![Build geniusai-server](https://github.com/LrGenius/geniusai-server/actions/workflows/build.yml/badge.svg)](https://github.com/LrGenius/geniusai-server/actions/workflows/build.yml)

### Development (uv)

```bash
uv sync
cp .env.postgre.example .env.postgre.local
./run.sh --dotenv .env.postgre.local --preload-models
```

macOS/Linux:

```bash
cp .env.postgre.example .env.postgre.local
./run.sh --fetch-models #populate cache with models
./run.sh --dotenv .env.postgre.local # load PostgreSQL and model cache settings from a git-ignored dotenv file
./run.sh --database-name <database_name> # use an explicit PostgreSQL database name
./run.sh --model-cache-path <model_cache_path> # override embedding model cache path
./run.sh --debug #Debug mode
./run.sh --debug-in-file <log_path> # write full debug logs, incoming HTTP requests, and LLM provider request payloads to this file
./run.sh --lazy-load-models # skip command-line model preload
./run.sh --preload-models #Load models at startup instead of waiting 1st request
./run.sh #load models from cache, assuming this is present
#./run.sh --dotenv ".env.postgre.local"
#./run.sh --database-name "llava-qwen3"
```

### Docker

Build the image, then start the container with Docker Compose:

```bash
./build-docker-image.sh
docker compose up
```

The compose file loads the prebuilt `geniusai-server:latest` image, mounts `./.env` into `/config` as read-only, and sets `GENIUSAI_DOTENV_FILENAME` so the entrypoint picks the dotenv file from that mounted directory.

Runtime arguments such as host, port, preload/debug flags, and model cache path are read from the selected dotenv file. The default compose port mapping is `19819:19819`, so `GENIUSAI_SERVER_PORT` in that file should remain `19819` unless you also update [docker-compose.yml](/Users/baptiste/workspace/geniusai-server/docker-compose.yml:1).

`GENIUSAI_DATA_DIR` in the selected dotenv controls where the server writes `lrgenius-server.log`, `lrgenius-server.pid`, and `lrgenius-server.OK`. For Docker, mount a writable volume to the same in-container path you set in `GENIUSAI_DATA_DIR`.

`GENIUSAI_UPLOAD_TEMP_DIR` controls where multipart uploads received by `/index` and `/index_by_reference` are temporarily staged while Flask parses the request. If unset, it defaults to `<GENIUSAI_DATA_DIR>/uploads-temp`.

Indexing requests must upload photos in `multipart/form-data` using repeated `image` file fields and matching `uuid` and `filename` fields. The capture date contract is `photo_date`: clients send the photo capture date in a `photo_date` field, and the backend stores that value as canonical metadata `photo_date`. Do not send legacy date fields such as `date_time` or `photos_date`. If `submit_date_time=true`, that same `photo_date` value is also provided to the metadata prompt. Additional form fields are preserved as metadata, so EXIF or other image metadata can be sent alongside the upload. Server-side file paths in the request body are no longer supported.

The embedding model cache path controls where the Hugging Face metadata text embedding model is stored and loaded from. It is passed to Hugging Face as `cache_dir`, so the model will be stored under that directory using Hugging Face's cache layout.
Command-line wrapper launches preload the embedding model before the server accepts requests. Plugin launches keep lazy loading unless they pass `--preload-models`.

### Database

The server stores metadata and vectors in PostgreSQL with the `pgvector` extension. On first use it connects through the configured PostgreSQL URL, creates the selected database when it does not exist, enables `pgvector`, and creates the `photo_metadata` table plus vector indexes.

Database selection is configuration-driven:

- PostgreSQL connection settings and the model cache path belong in `.env.postgre.local`, which is ignored by git. The Makefile uses this file through `LOCAL_DOTENV ?= .env.postgre.local`.
- The dotenv file supports database settings plus runtime flags such as `GENIUSAI_SERVER_HOST`, `GENIUSAI_SERVER_PORT`, `GENIUSAI_DATA_DIR`, `GENIUSAI_UPLOAD_TEMP_DIR`, `MODEL_CACHE_PATH`, `GENIUSAI_FETCH_MODELS`, `GENIUSAI_PRELOAD_MODELS`, `GENIUSAI_DEBUG`, and `GENIUSAI_DEBUG_IN_FILE`.
- `--database-name <name>` uses an explicit database.
- Switching database is done by passing a different `--database-name`.

### Search

Indexing embeds generated photo metadata, not the image pixels. The default embedding model is `Qwen/Qwen3-Embedding-0.6B`; search terms use the model's instruction-style query format, while indexed metadata documents are embedded as plain text.

`/search` accepts `min_pertinence_score` as a GET query parameter or POST JSON field. The value must be between `0` and `1`; higher values return fewer, more relevant semantic matches. The default is `0.35`. Exact metadata matches are accent-insensitive and are returned even if semantic scoring is unavailable.

By default `/search` returns only `ai_model`, `ai_rundate`, `distance`, `filename`, `match_type`, `pertinence_score`, `photo_date`, and `uuid`. Add `return_metadata=true` to include the stored metadata payload for each hit.

The `/index` pipeline stores the canonical search-facing fields automatically, including `filename`, `ai_model`, `ai_rundate`, and `photo_date` from the request `photo_date` field.

`/get` is a `POST` endpoint for fetching stored photos and their metadata/quality payloads. Send filters in the JSON body using direct fields or nested `filters`, `metadata`, and `quality` objects. Supported filters include `uuid`, `filename`, `ai_model`, `ai_rundate`, `photo_date`, `provider`, and any other stored metadata key. If the body is empty, `/get` returns every photo. The response contains `count` plus a `photos` array with each record's `metadata` and `quality`.

### Models selection
- is done at the client side
- Default is ollama -> Needs to have a ollama listening locally on 11434 port
- Cloud providers can receive their API key per request through `api_key`; `/models` also accepts provider-specific keys such as `openai_apikey`, `gemini_apikey`, `mistral_apikey`, and `anthropic_apikey`. When a key is provided, `/models` asks the provider API for available models. Providers return an empty list when no models are returned.
