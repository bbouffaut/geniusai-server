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
./run.sh #load models from cache, assuming this is present
```

### Models selection
- is done at the client side
- Default is ollama -> Needs to have a ollama listening locally on 11434 port