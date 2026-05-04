## geniusai-server

The python backend for LrGeniusAI and possibly others in the future...

[![Build geniusai-server](https://github.com/LrGenius/geniusai-server/actions/workflows/build.yml/badge.svg)](https://github.com/LrGenius/geniusai-server/actions/workflows/build.yml)

### Development (uv)

```bash
uv sync
uv run python src/geniusai_server.py --db-path <db_path> --preload-models
```

macOS/Linux:

```bash
./run.sh --fetch-models #populate cache with models
./run.sh --db-path <db_path> # specify Genius DB Path
./run.sh --model-cache-path <model_cache_path> # specify embedding model cache path
./run.sh --debug #Debug mode
./run.sh --debug-in-file <log_path> # write full debug logs and unredacted LLM request payloads to this file
./run.sh --lazy-load-models # skip command-line model preload
./run.sh --preload-models #Load models at startup instead of waiting 1st request
./run.sh #load models from cache, assuming this is present
#./run.sh --db-path "/Volumes/Extreme SSD/Lightroom Plugins/lrgeniusAI-data/lrgenius.db" --model-cache-path "/Volumes/Extreme SSD/Lightroom Plugins/lrgeniusAI-data/embeddings-models-cache/"
#./run.sh --fetch-models --db-path "/Volumes/Extreme SSD/Lightroom Plugins/lrgeniusAI-data/lrgenius.db" --model-cache-path "/Volumes/Extreme SSD/Lightroom Plugins/lrgeniusAI-data/embeddings-models-cache/"
```

The embedding model cache path controls where the Hugging Face metadata text embedding model is stored and loaded from. It is passed to Hugging Face as `cache_dir`, so the model will be stored under that directory using Hugging Face's cache layout.
Command-line wrapper launches preload the embedding model before the server accepts requests. Plugin launches keep lazy loading unless they pass `--preload-models`.

### Search

Indexing embeds generated photo metadata, not the image pixels. The default embedding model is `Qwen/Qwen3-Embedding-0.6B`; search terms use the model's instruction-style query format, while indexed metadata documents are embedded as plain text.

`/search` accepts `min_pertinence_score` as a GET query parameter or POST JSON field. The value must be between `0` and `1`; higher values return fewer, more relevant semantic matches. The default is `0.35`. Exact metadata matches are accent-insensitive and are returned even if semantic scoring is unavailable.

### Models selection
- is done at the client side
- Default is ollama -> Needs to have a ollama listening locally on 11434 port
- Cloud providers can receive their API key per request through `api_key`; `/models` also accepts provider-specific keys such as `openai_apikey`, `gemini_apikey`, and `mistral_apikey`. When a key is provided, `/models` asks the provider API for available models and falls back to the built-in list if listing is unavailable.
