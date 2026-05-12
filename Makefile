RUN_SCRIPT := ./run.sh
DEBUG-FILE ?= ./output.log
DATABASE_NAME ?=
DATABASE_NAME_FLAG := $(if $(strip $(DATABASE_NAME)),--database-name "$(DATABASE_NAME)",)
DOTENV_LOCAL ?= .env/.env.postgre.local
DOTENV_SSD ?= .env/.env.postgre.ssd
DOTENV_PROD ?= .env/.env.postgre.prod

.PHONY: dev dev-local dev-debug-in-file prod

dev-ssd:
	KMP_DUPLICATE_LIB_OK=TRUE $(RUN_SCRIPT) --dotenv $(DOTENV_SSD) $(DATABASE_NAME_FLAG) --debug --preload-models

dev-local:
	KMP_DUPLICATE_LIB_OK=TRUE $(RUN_SCRIPT) --dotenv $(DOTENV_LOCAL) $(DATABASE_NAME_FLAG) --debug --preload-models

dev-local-fetch-models:
	KMP_DUPLICATE_LIB_OK=TRUE $(RUN_SCRIPT) --dotenv $(DOTENV_LOCAL) $(DATABASE_NAME_FLAG) --debug --preload-models --fetch-models

dev-debug-in-file:
	KMP_DUPLICATE_LIB_OK=TRUE $(RUN_SCRIPT) --dotenv $(DOTENV_SSD) $(DATABASE_NAME_FLAG) --debug-in-file ${DEBUG-FILE}  --preload-models

prod:
	$(RUN_SCRIPT) --dotenv $(DOTENV_PROD) $(DATABASE_NAME_FLAG) --preload-models --fetch-models
