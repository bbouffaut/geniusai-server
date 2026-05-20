RUN_SCRIPT := ./run.sh
DOTENV_LOCAL ?= .env/.env.postgre.local
DOTENV_LOCAL_POSTGRE_PROD ?= .env/.env.local.postgre.prod
DOTENV_SSD ?= .env/.env.postgre.ssd
DOTENV_PROD ?= .env/.env.postgre.prod

.PHONY: dev dev-local dev-debug-in-file prod

dev-ssd:
	KMP_DUPLICATE_LIB_OK=TRUE $(RUN_SCRIPT) --dotenv $(DOTENV_SSD)

dev-local:
	KMP_DUPLICATE_LIB_OK=TRUE $(RUN_SCRIPT) --dotenv $(DOTENV_LOCAL)

dev-local-postgre-prod:
	KMP_DUPLICATE_LIB_OK=TRUE $(RUN_SCRIPT) --dotenv $(DOTENV_LOCAL_POSTGRE_PROD)

prod:
	$(RUN_SCRIPT) --dotenv $(DOTENV_PROD)
