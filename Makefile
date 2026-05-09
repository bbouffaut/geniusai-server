RUN_SCRIPT := ./run.sh
DEBUG-FILE ?= ./output.log
DATABASE_NAME ?= geniusai-server_qwen3-embedding-0.6b
DOTENV_LOCAL ?= .env.postgre.local
DOTENV_SSD ?= .env.postgre.ssd
DOTENV_PROD ?= .env.postgre.prod

.PHONY: dev dev-local dev-debug-in-file prod

dev-ssd:
	KMP_DUPLICATE_LIB_OK=TRUE $(RUN_SCRIPT) --dotenv $(DOTENV_SSD) --database-name "$(DATABASE_NAME)" --debug --preload-models

dev-local:
	KMP_DUPLICATE_LIB_OK=TRUE $(RUN_SCRIPT) --dotenv $(DOTENV_LOCAL) --database-name "$(DATABASE_NAME)" --debug --preload-models

dev-debug-in-file:
	KMP_DUPLICATE_LIB_OK=TRUE $(RUN_SCRIPT) --dotenv $(DOTENV_SSD) --database-name "$(DATABASE_NAME)" --debug-in-file ${DEBUG-FILE}  --preload-models

prod:
	$(RUN_SCRIPT) --dotenv $(DOTENV_PROD) --database-name "$(DATABASE_NAME)" --preload-models
