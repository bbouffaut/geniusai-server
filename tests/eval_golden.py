"""T3.2 — golden-set evaluation of /style-match/search.

Labels are APPLY-WORTHY matches (photos whose settings you would actually
apply), not merely the visually closest — recall@5 is computed against that
criterion (plan §3.8).

    PYTHONPATH=src uv run python tests/eval_golden.py --server-url http://rpi20:19819 \
        --golden tests/golden_set.json [--neutral-dir <dir>]

For every golden query it runs the search in both scoring modes (linear, rrf)
and both has_edits settings, reports recall@5, and dumps per-signal score
distributions so the 50/30/10/10 weights can be sanity-checked (plan §2.3).
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import requests

SEARCH_ENDPOINT = "/style-match/search"


def load_golden(path):
    data = json.loads(Path(path).read_text())
    queries = data.get("queries") if isinstance(data, dict) else data
    if not isinstance(queries, list):
        raise SystemExit("Golden set must be {\"queries\": [...]} or a JSON array.")
    for entry in queries:
        if "query" not in entry or "expected_top5" not in entry:
            raise SystemExit(
                "Each golden entry needs 'query' (path or uuid) and 'expected_top5' (list of paths)."
            )
    return queries


def run_search(server_url, image_path, neutral_path, top_k, mode, has_edits, timeout=120):
    data = {
        "top_k": str(top_k),
        "scoring_mode": mode,
        "filters": json.dumps({"has_edits": has_edits}),
    }
    files = {"image": (Path(image_path).name, Path(image_path).read_bytes(), "image/jpeg")}
    if neutral_path and Path(neutral_path).is_file():
        files["neutral_image"] = (Path(neutral_path).name, Path(neutral_path).read_bytes(), "image/jpeg")
    response = requests.post(
        server_url.rstrip("/") + SEARCH_ENDPOINT, data=data, files=files, timeout=timeout
    )
    response.raise_for_status()
    return response.json()


def evaluate(queries, server_url, top_k=5):
    report = {}
    all_signals = {mode: {"sem": [], "desc": [], "color": [], "loc": []} for mode in ("linear", "rrf")}
    recalls = {mode: {} for mode in ("linear", "rrf")}

    for mode in ("linear", "rrf"):
        for has_edits in (True, False):
            key = f"{mode}/has_edits={'on' if has_edits else 'off'}"
            hits = 0
            evaluated = 0
            latencies = []
            for entry in queries:
                try:
                    payload = run_search(
                        server_url,
                        entry["query"],
                        entry.get("neutral"),
                        top_k,
                        mode,
                        has_edits,
                    )
                except Exception as error:
                    print(f"SEARCH FAILED for {entry['query']} [{key}]: {error}", file=sys.stderr)
                    continue
                latencies.append(payload.get("latency_ms") or 0)
                expected = set(Path(p).name for p in entry["expected_top5"])
                returned = [match.get("file_name") or Path(match.get("file_path") or "").name for match in payload["matches"][:top_k]]
                if expected & set(returned):
                    hits += 1
                evaluated += 1
                for match in payload["matches"][:top_k]:
                    for signal, value in match.get("signals", {}).items():
                        if value is not None:
                            all_signals[mode][signal].append(value)
            recalls[mode][key] = {
                "recall@5": round(hits / evaluated, 3) if evaluated else None,
                "queries": evaluated,
                "median_latency_ms": round(statistics.median(latencies)) if latencies else None,
            }

    report["recalls"] = recalls
    report["signal_distributions"] = {
        mode: {
            signal: {
                "n": len(values),
                "mean": round(statistics.mean(values), 3) if values else None,
                "p10": round(np_percentile(values, 10), 3) if values else None,
                "p90": round(np_percentile(values, 90), 3) if values else None,
            }
            for signal, values in signals.items()
        }
        for mode, signals in all_signals.items()
    }
    return report


def np_percentile(values, percentile):
    values = sorted(values)
    if not values:
        return None
    index = min(len(values) - 1, max(0, int(round((percentile / 100.0) * (len(values) - 1)))))
    return values[index]


def main():
    parser = argparse.ArgumentParser(description="Golden-set style-match evaluation")
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--golden", default=str(Path(__file__).parent / "golden_set.json"))
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    queries = load_golden(args.golden)
    if not queries:
        print(
            "Golden set is empty — label 20–30 photos whose settings you would actually "
            "apply (plan §3.8 / task T0.3), then re-run this script.",
            file=sys.stderr,
        )
        return 7  # reserved exit code: prerequisite data missing

    report = evaluate(queries, args.server_url, args.top_k)
    print(json.dumps(report, indent=2))

    best = None
    for mode in ("linear", "rrf"):
        for key, stats in report["recalls"][mode].items():
            recall = stats["recall@5"]
            if recall is not None and (best is None or recall > best[1]):
                best = (key, recall)
    if best:
        print(f"\nBest configuration: {best[0]} recall@5={best[1]}")
        if best[1] < 0.8:
            print("BELOW the 0.8 target — inspect per-signal distributions above; "
                  "consider the RRF fallback or weight tuning (plan §2.3).", file=sys.stderr)
            return 3  # validation failed
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
