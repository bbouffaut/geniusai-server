DB_PATH ?= /Volumes/Extreme SSD/Lightroom Plugins/lrgeniusAI-data/lrgenius.db
MODEL_CACHE_PATH ?= /Volumes/Extreme SSD/Lightroom Plugins/lrgeniusAI-data/embeddings-models-cache/
MODEL_CACHE_ARG := $(if $(MODEL_CACHE_PATH),--model-cache-path "$(MODEL_CACHE_PATH)",)
RUN_SCRIPT := ./run.sh

.PHONY: dev prod

dev:
	KMP_DUPLICATE_LIB_OK=TRUE uv run python src/geniusai_server.py --db-path "$(DB_PATH)" --debug --preload-models $(MODEL_CACHE_ARG)

prod:
	$(RUN_SCRIPT) --db-path "$(DB_PATH)" --preload-models $(MODEL_CACHE_ARG)
