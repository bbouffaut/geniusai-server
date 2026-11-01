"""Shared helper for tests that build a minimal fake `config` module.

Style-match code imports additional config names; tests that stub config with
only the legacy names call `apply_style_match_config` to fill the defaults.
"""


def apply_style_match_config(fake_config):
    """Add the style-match config defaults to a fake config module in place."""
    fake_config.CLIP_EMBEDDING_DIMENSION = 512
    fake_config.DEBUG_LEVEL = 0
    fake_config.DEFAULT_METADATA_LANGUAGE = "English"
    fake_config.DEFAULT_METADATA_PROVIDER = "ollama"
    fake_config.STYLE_MATCH_CANDIDATE_K = 200
    fake_config.STYLE_MATCH_DEFAULT_WEIGHTS = {"sem": 0.5, "desc": 0.3, "color": 0.1, "loc": 0.1}
    fake_config.STYLE_MATCH_EMBEDDING_MODEL = "clip-vit-b32"
    fake_config.STYLE_MATCH_ENABLED = True
    fake_config.STYLE_MATCH_LOC_TAU_KM = 50.0
    fake_config.STYLE_MATCH_MAX_ORIGINAL_BYTES = 200 * 1024 * 1024
    fake_config.STYLE_MATCH_SCORING_MODE = "linear"
    return fake_config
