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
./run.sh --data-dir <data_dir> # specify runtime log/pid directory
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

The embedding model cache path controls where the Hugging Face metadata text embedding model is stored and loaded from. It is passed to Hugging Face as `cache_dir`, so the model will be stored under that directory using Hugging Face's cache layout.
Command-line wrapper launches preload the embedding model before the server accepts requests. Plugin launches keep lazy loading unless they pass `--preload-models`.

### Database

The server stores metadata and vectors in PostgreSQL with the `pgvector` extension. On first use it connects through the configured PostgreSQL URL, creates the selected database when it does not exist, enables `pgvector`, and creates the `photo_metadata` table plus vector indexes.

Database selection is configuration-driven:

- PostgreSQL connection settings and the model cache path belong in `.env.postgre.local`, which is ignored by git. The Makefile uses this file through `LOCAL_DOTENV ?= .env.postgre.local`.
- The dotenv file supports `GENIUSAI_POSTGRES_URL`, `GENIUSAI_POSTGRES_USER`, `GENIUSAI_POSTGRES_PASSWORD`, and `MODEL_CACHE_PATH`.
- `--database-name <name>` uses an explicit database.
- Switching database is done by passing a different `--database-name`.

### Search

Indexing embeds generated photo metadata, not the image pixels. The default embedding model is `Qwen/Qwen3-Embedding-0.6B`; search terms use the model's instruction-style query format, while indexed metadata documents are embedded as plain text.

`/search` accepts `min_pertinence_score` as a GET query parameter or POST JSON field. The value must be between `0` and `1`; higher values return fewer, more relevant semantic matches. The default is `0.35`. Exact metadata matches are accent-insensitive and are returned even if semantic scoring is unavailable.

### Models selection
- is done at the client side
- Default is ollama -> Needs to have a ollama listening locally on 11434 port
- Cloud providers can receive their API key per request through `api_key`; `/models` also accepts provider-specific keys such as `openai_apikey`, `gemini_apikey`, `mistral_apikey`, and `anthropic_apikey`. When a key is provided, `/models` asks the provider API for available models. Providers return an empty list when no models are returned.
