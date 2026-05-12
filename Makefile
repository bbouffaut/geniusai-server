RUN_SCRIPT := ./run.sh
DOTENV_LOCAL ?= .env/.env.postgre.local
DOTENV_SSD ?= .env/.env.postgre.ssd
DOTENV_PROD ?= .env/.env.postgre.prod

.PHONY: dev dev-local dev-debug-in-file prod

dev-ssd:
	KMP_DUPLICATE_LIB_OK=TRUE $(RUN_SCRIPT) --dotenv $(DOTENV_SSD)

dev-local:
	KMP_DUPLICATE_LIB_OK=TRUE $(RUN_SCRIPT) --dotenv $(DOTENV_LOCAL)

dev-local-fetch-models:
	KMP_DUPLICATE_LIB_OK=TRUE $(RUN_SCRIPT) --dotenv $(DOTENV_LOCAL)

dev-debug-in-file:
	KMP_DUPLICATE_LIB_OK=TRUE $(RUN_SCRIPT) --dotenv $(DOTENV_SSD)

prod:
	$(RUN_SCRIPT) --dotenv $(DOTENV_PROD)
