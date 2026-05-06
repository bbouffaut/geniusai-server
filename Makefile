MODEL_CACHE_PATH ?= /Volumes/Extreme SSD/Lightroom Plugins/lrgeniusAI-data/embeddings-models-cache/
MODEL_CACHE_ARG := $(if $(MODEL_CACHE_PATH),--model-cache-path "$(MODEL_CACHE_PATH)",)
RUN_SCRIPT := ./run.sh
DEBUG-FILE ?= ./output.log
DATABASE_NAME ?= qwen3-embedding-0.6b

.PHONY: dev prod

dev:
	KMP_DUPLICATE_LIB_OK=TRUE $(RUN_SCRIPT) --database-name "$(DATABASE_NAME)" --debug --preload-models $(MODEL_CACHE_ARG)

dev-local:
	KMP_DUPLICATE_LIB_OK=TRUE $(RUN_SCRIPT) --database-name "$(DATABASE_NAME)" --debug --preload-models --model-cache-path ./cache/

dev-debug-in-file:
	KMP_DUPLICATE_LIB_OK=TRUE $(RUN_SCRIPT) --database-name "$(DATABASE_NAME)" --debug-in-file ${DEBUG-FILE}  --preload-models $(MODEL_CACHE_ARG)

prod:
	$(RUN_SCRIPT) --database-name "$(DATABASE_NAME)" --preload-models $(MODEL_CACHE_ARG)
