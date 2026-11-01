import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from style_match import scoring


def test_renormalization_drops_missing_signals():
    total, weights = scoring.linear_score(
        {"sem": 0.8, "desc": 0.6},  # no color/loc
        {"sem": 0.5, "desc": 0.3, "color": 0.1, "loc": 0.1},
    )
    assert abs(weights["sem"] - 0.5 / 0.8) < 1e-9
    assert abs(weights["desc"] - 0.3 / 0.8) < 1e-9
    assert abs(total - (0.625 * 0.8 + 0.375 * 0.6)) < 1e-9


def test_linear_score_full_weights():
    total, weights = scoring.linear_score(
        {"sem": 1.0, "desc": 1.0, "color": 1.0, "loc": 1.0}, scoring.DEFAULT_WEIGHTS
    )
    assert abs(total - 1.0) < 1e-9
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_all_signals_missing_scores_zero():
    total, weights = scoring.linear_score({}, scoring.DEFAULT_WEIGHTS)
    assert total == 0.0
    assert weights == {}


def test_rrf_favors_top_ranks():
    # Candidate A: rank 1 everywhere; candidate B: rank 2 everywhere.
    score_a, _ = scoring.rrf_score(
        {"sem": 1, "desc": 1}, {"sem": 0.5, "desc": 0.3, "color": 0.1, "loc": 0.1}
    )
    score_b, weights = scoring.rrf_score(
        {"sem": 2, "desc": 2}, {"sem": 0.5, "desc": 0.3, "color": 0.1, "loc": 0.1}
    )
    assert score_a > score_b
    # Missing colour/loc weights are renormalized into the remaining two.
    assert abs(weights["sem"] + weights["desc"] - 1.0) < 1e-9


def test_rank_candidates_ranks_per_signal_and_skips_missing():
    candidates = [
        {"sem": 0.9, "desc": None, "color": 0.5, "loc": None},
        {"sem": 0.8, "desc": 0.9, "color": 0.4, "loc": None},
        {"sem": 0.7, "desc": 0.8, "color": 0.6, "loc": None},
    ]
    ranks = scoring.rank_candidates(candidates)
    assert ranks["sem"] == {0: 1, 1: 2, 2: 3}
    assert ranks["desc"] == {1: 1, 2: 2}
    assert ranks["color"] == {2: 1, 0: 2, 1: 3}
    assert ranks["loc"] == {}


def test_rrf_mode_via_score_dispatch():
    total, _ = scoring.score(ranks={"sem": 1}, weights={"sem": 1.0}, mode="rrf")
    assert abs(total - 1.0 / (1.0 + scoring.RRF_K)) < 1e-9
