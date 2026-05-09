RUN_SCRIPT := ./run.sh
DEBUG-FILE ?= ./output.log
DATABASE_NAME ?= geniusai-server_qwen3-embedding-0.6b
LOCAL_DOTENV ?= .env.postgre.local
LOCAL_DOTENV_ARG := $(if $(LOCAL_DOTENV),--dotenv "$(LOCAL_DOTENV)",)

.PHONY: dev dev-local dev-debug-in-file prod

dev:
	KMP_DUPLICATE_LIB_OK=TRUE $(RUN_SCRIPT) $(LOCAL_DOTENV_ARG) --database-name "$(DATABASE_NAME)" --debug --preload-models

dev-local:
	KMP_DUPLICATE_LIB_OK=TRUE $(RUN_SCRIPT) $(LOCAL_DOTENV_ARG) --database-name "$(DATABASE_NAME)" --debug --preload-models

dev-debug-in-file:
	KMP_DUPLICATE_LIB_OK=TRUE $(RUN_SCRIPT) $(LOCAL_DOTENV_ARG) --database-name "$(DATABASE_NAME)" --debug-in-file ${DEBUG-FILE}  --preload-models

prod:
	$(RUN_SCRIPT) $(LOCAL_DOTENV_ARG) --database-name "$(DATABASE_NAME)" --preload-models
