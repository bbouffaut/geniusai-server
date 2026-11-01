"""Style-match scoring: hybrid weighted sum (linear) and weighted RRF.

The hybrid score mixes four signals:
  sem   – CLIP image-embedding cosine similarity (topic)
  desc  – description-embedding cosine similarity (existing Qwen3 pipeline)
  color – cosine similarity of HSV histograms (neutral renders, see plan §2.4)
  loc   – exp(-(haversine_km / tau)^2) GPS proximity

Missing-signal handling: when a signal is unavailable (no description yet, no
neutral render, no GPS on either side), its weight is dropped and the remaining
weights are renormalized to sum to 1. Per-signal scores are always returned so
the ranking stays auditable.

Linear vs RRF (plan §2.3): the linear sum mixes signals with different
similarity ranges (CLIP cosine is compressed toward the top of its range while
loc_sim spans [0, 1]). The drop-in fallback is weighted reciprocal-rank fusion
`score = sum(w_s / (60 + rank_s))`, computed per signal from the candidate's
rank within that signal-ordered candidate list. Both modes are selectable so
they can be measured against the golden set (T3.2) instead of chosen by vibes.
"""

RRF_K = 60  # standard reciprocal-rank-fusion constant

DEFAULT_WEIGHTS = {"sem": 0.5, "desc": 0.3, "color": 0.1, "loc": 0.1}


def renormalize_weights(weights, available_signals):
    """Return the weights restricted to `available_signals`, renormalized to 1.

    Signals with a zero or missing weight are treated as unavailable.
    """
    available = [key for key in ("sem", "desc", "color", "loc") if key in available_signals]
    picked = {}
    for key in available:
        weight = float(weights.get(key, 0.0) or 0.0)
        if weight > 0:
            picked[key] = weight
    total = sum(picked.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in picked.items()}


def linear_score(signals, weights):
    """Weighted sum over the available signals.

    Returns (total_score, weights_used). `signals` maps signal name -> value
    in [0, 1]; unavailable signals must be omitted (not zero).
    """
    weights_used = renormalize_weights(weights, signals.keys())
    if not weights_used:
        return 0.0, {}
    total = sum(weights_used[key] * float(signals[key]) for key in weights_used)
    return total, weights_used


def rrf_score(ranks, weights):
    """Weighted reciprocal-rank fusion.

    `ranks` maps signal name -> 1-based rank of the candidate within that
    signal-ordered candidate list (missing signal -> omitted). Ties must be
    resolved by the caller (e.g. stable sort order).
    Returns (total_score, weights_used).
    """
    weights_used = renormalize_weights(weights, ranks.keys())
    if not weights_used:
        return 0.0, {}
    total = sum(weights_used[key] / (RRF_K + float(ranks[key])) for key in weights_used)
    return total, weights_used


def score(signals=None, ranks=None, weights=None, mode="linear"):
    """Score one candidate; returns (total, weights_used).

    mode "linear" uses `signals` (values), mode "rrf" uses `ranks`.
    """
    weights = weights or DEFAULT_WEIGHTS
    if mode == "rrf":
        return rrf_score(ranks or {}, weights)
    return linear_score(signals or {}, weights)


def rank_candidates(candidates):
    """Return {candidate_position: 1-based rank} per signal.

    candidates: list of dicts containing per-signal values under their name
    ("sem", "desc", "color", "loc"); candidates lacking a signal value are
    excluded from that signal's ranking.
    Returns {signal: {candidate_position: rank}}.
    """
    ranks = {}
    for signal in ("sem", "desc", "color", "loc"):
        ordered = [
            position
            for position, candidate in enumerate(candidates)
            if candidate.get(signal) is not None
        ]
        ordered.sort(key=lambda position: -float(candidates[position].get(signal) or 0.0))
        ranks[signal] = {position: rank + 1 for rank, position in enumerate(ordered)}
    return ranks
