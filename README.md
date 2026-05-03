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
./run.sh --lazy-load-models # skip command-line model preload
./run.sh --preload-models #Load models at startup instead of waiting 1st request
./run.sh #load models from cache, assuming this is present
#./run.sh --db-path "/Volumes/Extreme SSD/Lightroom Plugins/lrgeniusAI-data/lrgenius.db" --model-cache-path "/Volumes/Extreme SSD/Lightroom Plugins/lrgeniusAI-data/embeddings-models-cache/"
#./run.sh --fetch-models --db-path "/Volumes/Extreme SSD/Lightroom Plugins/lrgeniusAI-data/lrgenius.db" --model-cache-path "/Volumes/Extreme SSD/Lightroom Plugins/lrgeniusAI-data/embeddings-models-cache/"
```

The embedding model cache path controls where the OpenCLIP/SigLIP2 model is stored and loaded from. It is passed to Hugging Face as `cache_dir`, so the model will be stored under that directory using Hugging Face's cache layout.
Command-line wrapper launches preload the embedding model before the server accepts requests. Plugin launches keep lazy loading unless they pass `--preload-models`.

### Search

`/search` accepts `min_pertinence_score` as a GET query parameter or POST JSON field. The value must be between `0` and `1`; higher values return fewer, more relevant semantic matches. The default is `0.2`.

### Models selection
- is done at the client side
- Default is ollama -> Needs to have a ollama listening locally on 11434 port
- Cloud providers can receive their API key per request through `api_key`; `/models` also accepts provider-specific keys such as `openai_apikey`, `gemini_apikey`, and `mistral_apikey`.
