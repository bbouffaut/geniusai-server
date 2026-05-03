DB_PATH ?= /Volumes/Extreme SSD/Lightroom Plugins/lrgeniusAI-data/lrgenius.db
MODEL_CACHE_PATH ?= 
MODEL_CACHE_ARG := $(if $(MODEL_CACHE_PATH),--model-cache-path "$(MODEL_CACHE_PATH)",)
RUN_SCRIPT := ./run.sh

.PHONY: dev prod

dev:
	KMP_DUPLICATE_LIB_OK=TRUE uv run python src/geniusai_server.py --db-path "$(DB_PATH)" --debug $(MODEL_CACHE_ARG)

prod:
	$(RUN_SCRIPT) --db-path "$(DB_PATH)" $(MODEL_CACHE_ARG)
