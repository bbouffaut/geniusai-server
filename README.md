## geniusai-server

The python backend for LrGeniusAI and possibly others in the future...

[![Build geniusai-server](https://github.com/LrGenius/geniusai-server/actions/workflows/build.yml/badge.svg)](https://github.com/LrGenius/geniusai-server/actions/workflows/build.yml)

### Development (uv)

```bash
uv sync
uv run python src/geniusai_server.py
```

macOS/Linux:

```bash
./run.sh --fetch-models #populate cache with models
./run.sh --db-path <db_path> # specify Genius DB Path
./run.sh --model-cache-path <model_cache_path> # specify embedding model cache path
./run.sh --debug #Debug mode
./run.sh #load models from cache, assuming this is present
#./run.sh --db-path "/Volumes/Extreme SSD/Lightroom Plugins/lrgeniusAI-data/lrgenius.db" --model-cache-path "/Volumes/Extreme SSD/Lightroom Plugins/lrgeniusAI-data/embeddings-models-cache/"
#./run.sh --fetch-models --db-path "/Volumes/Extreme SSD/Lightroom Plugins/lrgeniusAI-data/lrgenius.db" --model-cache-path "/Volumes/Extreme SSD/Lightroom Plugins/lrgeniusAI-data/embeddings-models-cache/"
```

The embedding model cache path controls where the OpenCLIP/SigLIP2 model is stored and loaded from. It is passed to Hugging Face as `cache_dir`, so the model will be stored under that directory using Hugging Face's cache layout.

### Models selection
- is done at the client side
- Default is ollama -> Needs to have a ollama listening locally on 11434 port
- Cloud providers can receive their API key per request through `api_key`; `/models` also accepts provider-specific keys such as `openai_apikey`, `gemini_apikey`, and `mistral_apikey`.
