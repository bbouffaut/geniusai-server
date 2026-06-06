RUN_SCRIPT     := ./run.sh
MIGRATE_SCRIPT := ./migrate.sh
DOTENV_LOCAL ?= config/.env.postgre.local
DOTENV_LOCAL_POSTGRE_PROD ?= config/.env.prod
DOTENV_SSD ?= config/.env.postgre.ssd
DOTENV_PROD ?= config/.env.postgre.prod

# Migration variables — override on the command line:
#   make migrate TARGET_DB=mydb-bge TARGET_MODEL=bge-m3 [DOTENV=<env_file>]
# Source DB is read from GENIUSAI_DATABASE_NAME in the dotenv file.
TARGET_DB    ?=
TARGET_MODEL ?=
DOTENV       ?= $(DOTENV_LOCAL)

.PHONY: dev dev-local dev-debug-in-file prod migrate

dev-ssd:
	KMP_DUPLICATE_LIB_OK=TRUE $(RUN_SCRIPT) --dotenv $(DOTENV_SSD)

dev-local:
	KMP_DUPLICATE_LIB_OK=TRUE $(RUN_SCRIPT) --dotenv $(DOTENV_LOCAL)

dev-local-postgre-prod:
	KMP_DUPLICATE_LIB_OK=TRUE $(RUN_SCRIPT) --dotenv $(DOTENV_LOCAL_POSTGRE_PROD)

prod:
	$(RUN_SCRIPT) --dotenv $(DOTENV_PROD)

## Re-index photo embeddings into TARGET_DB using TARGET_MODEL.
## Source DB is read from GENIUSAI_DATABASE_NAME in the dotenv file.
## Usage: make migrate TARGET_DB=<tgt> TARGET_MODEL=<model> [DOTENV=<env_file>]
## Available models: qwen3-0.6b, bge-m3
migrate:
	@[ -n "$(TARGET_DB)" ]    || (echo "Error: TARGET_DB is required.    Usage: make migrate TARGET_DB=<tgt> TARGET_MODEL=<model>" && exit 1)
	@[ -n "$(TARGET_MODEL)" ] || (echo "Error: TARGET_MODEL is required." && exit 1)
	KMP_DUPLICATE_LIB_OK=TRUE $(MIGRATE_SCRIPT) \
		--dotenv       $(DOTENV) \
		--target-db    $(TARGET_DB) \
		--target-model $(TARGET_MODEL)
