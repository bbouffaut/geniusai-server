DB_PATH ?= /Volumes/Extreme SSD/Lightroom Plugins/lrgeniusAI-data/lrgenius.db
RUN_SCRIPT := ./run.sh

.PHONY: dev prod

dev:
	KMP_DUPLICATE_LIB_OK=TRUE uv run python src/geniusai_server.py --db-path "$(DB_PATH)" --debug

prod:
	$(RUN_SCRIPT) --db-path "$(DB_PATH)"
