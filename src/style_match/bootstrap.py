"""T2.2 — Style-Match bootstrap: upload the plugin manifest to the server.

Runs on the user's machine (the k3s-rpi20 server has no filesystem access to
the catalog — the upload contract is kept, plan §2.7):

    PYTHONPATH=src uv run python -m style_match.bootstrap \
        --manifest manifest.json --server-url http://rpi20:19819 \
        --catalog my-catalog --batch 8 --concurrency 2

Per photo it resolves two renders (plan §2.4):
  - current look  <- the manifest's preview_path (mandatory for edited photos)
  - neutral       <- RAW embedded JPEG via rawpy, or the JPEG original itself

Resume + idempotency: `.bootstrap-progress.json` caches (mtime, size) per
lrc_local_id so unchanged photos are skipped; the server upserts by
(catalog_id, lrc_local_id), so re-uploads never duplicate rows.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

RAW_SUFFIXES = {".arw", ".cr2", ".cr3", ".nef", ".nrw", ".orf", ".raf", ".rw2", ".dng", ".pef", ".srw"}
JPEG_SUFFIXES = {".jpg", ".jpeg", ".jpe", ".png", ".webp", ".tif", ".tiff"}

INDEX_ENDPOINT = "/style-match/index"


def resolve_current_look(entry):
    """Current-look JPEG bytes from the manifest preview (plan §2.4)."""
    preview_path = entry.get("preview_path")
    if not preview_path:
        return None, "missing preview_path"
    path = Path(preview_path)
    if not path.is_file():
        return None, f"preview not found: {preview_path}"
    return path.read_bytes(), None


def resolve_neutral_render(file_path):
    """Neutral render for the HSV histogram: RAW embedded JPEG or the file."""
    path = Path(file_path)
    if not path.is_file():
        return None
    suffix = path.suffix.lower()
    if suffix in JPEG_SUFFIXES:
        return path.read_bytes()
    if suffix in RAW_SUFFIXES:
        try:
            import rawpy
        except ImportError:
            print("rawpy not installed — RAW neutral renders skipped "
                  "(colour signal stays NULL until backfill).", file=sys.stderr)
            return None
        try:
            with rawpy.imread(str(path)) as raw:
                return raw.extract_embedded_preview()
        except Exception as error:
            print(f"Embedded preview failed for {path.name}: {error}", file=sys.stderr)
            return None
    return None


def load_progress(progress_path):
    if progress_path.is_file():
        try:
            return json.loads(progress_path.read_text())
        except (ValueError, OSError):
            pass
    return {"processed": {}}


def save_progress(progress_path, progress):
    progress_path.write_text(json.dumps(progress))


def _fingerprint(path):
    stat = path.stat()
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def upload_entry(server_url, catalog_id, entry, request_timeout=120):
    image_bytes, error = resolve_current_look(entry)
    if image_bytes is None:
        return entry, False, error

    lrc_local_id = entry.get("lrc_local_id")
    if lrc_local_id is None:
        return entry, False, "missing lrc_local_id"

    neutral_bytes = resolve_neutral_render(entry.get("file_path") or "")
    data = {
        "uuid": entry.get("uuid") or "",
        "filename": entry.get("file_name") or Path(entry.get("file_path") or "").name,
        "capture_time": entry.get("capture_time") or "",
        "catalog_id": catalog_id,
        "lrc_local_id": str(lrc_local_id),
        "file_path": entry.get("file_path") or "",
        "latitude": "" if entry.get("latitude") is None else entry["latitude"],
        "longitude": "" if entry.get("longitude") is None else entry["longitude"],
        "camera_make": entry.get("camera_make") or "",
        "camera_model": entry.get("camera_model") or "",
        "lens": entry.get("lens") or "",
        "focal_length_mm": "" if entry.get("focal_length_mm") is None else entry["focal_length_mm"],
        "aperture_f_number": "" if entry.get("aperture_f_number") is None else entry["aperture_f_number"],
        "iso": "" if entry.get("iso") is None else entry["iso"],
        "has_edits": "true" if entry.get("has_edits") else "false",
        "generate_description": "false" if entry.get("description") else "true",
        "render_source": "develop_export",
    }
    if entry.get("description"):
        data["description"] = entry["description"]
    if entry.get("develop_settings") is not None:
        data["develop_settings"] = json.dumps(entry["develop_settings"])

    files = {"image": ("current.jpg", image_bytes, "image/jpeg")}
    if neutral_bytes:
        files["neutral_image"] = ("neutral.jpg", neutral_bytes, "image/jpeg")

    last_error = None
    for attempt in range(3):
        try:
            response = requests.post(
                server_url.rstrip("/") + INDEX_ENDPOINT,
                data=data,
                files=files,
                timeout=request_timeout,
            )
            if response.status_code == 200:
                payload = response.json()
                return entry, True, payload
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
        except requests.RequestException as error:
            last_error = str(error)
        time.sleep(1.5 * (attempt + 1))
    return entry, False, last_error


def main():
    parser = argparse.ArgumentParser(description="Style-Match bootstrap uploader")
    parser.add_argument("--manifest", required=True, help="Manifest JSON exported by the plugin")
    parser.add_argument("--server-url", required=True, help="e.g. http://rpi20:19819")
    parser.add_argument("--catalog", required=True, help="Catalog identifier used at index time")
    parser.add_argument("--batch", type=int, default=8, help="Reporting batch size")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--progress-file", default=".bootstrap-progress.json")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N rows (dry run)")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    entries = json.loads(manifest_path.read_text())
    if not isinstance(entries, list):
        print("Manifest must be a JSON array of photo rows.", file=sys.stderr)
        return 2
    if args.limit:
        entries = entries[: args.limit]

    progress_path = Path(args.progress_file)
    progress = load_progress(progress_path)
    processed = progress.setdefault("processed", {})

    pending = []
    skipped = 0
    for entry in entries:
        lrc_id = str(entry.get("lrc_local_id"))
        source = entry.get("file_path") or ""
        fingerprint = _fingerprint(Path(source)) if source and Path(source).is_file() else None
        cached = processed.get(lrc_id)
        if cached and (fingerprint is None or cached == fingerprint):
            skipped += 1
            continue
        pending.append(entry)

    print(f"Manifest rows: {len(entries)} | already indexed (resume cache): {skipped} | to upload: {len(pending)}")

    success = 0
    failures = []
    done = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = [
            executor.submit(upload_entry, args.server_url, args.catalog, entry, args.timeout)
            for entry in pending
        ]
        for future in as_completed(futures):
            entry, ok, result = future.result()
            done += 1
            if ok:
                success += 1
                source = entry.get("file_path") or ""
                fingerprint = _fingerprint(Path(source)) if source and Path(source).is_file() else None
                if fingerprint:
                    processed[str(entry["lrc_local_id"])] = fingerprint
                if done % args.batch == 0 or done == len(pending):
                    save_progress(progress_path, progress)
                    rate = done / max(time.time() - start, 1e-9)
                    print(f"[{done}/{len(pending)}] ok={success} failed={len(failures)} ({rate:.2f} photos/s)")
            else:
                failures.append({"lrc_local_id": entry.get("lrc_local_id"), "error": str(result)})
                print(f"FAILED {entry.get('lrc_local_id')}: {result}", file=sys.stderr)

    save_progress(progress_path, progress)
    print(f"Done: {success} indexed, {len(failures)} failed, {skipped} skipped (already cached).")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
