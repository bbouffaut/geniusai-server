"""
Background migration service.

Runs src/migrate.py as a subprocess so the target embedding model is loaded
independently from the model already in use by the running server.  The server
process is never interrupted and can keep serving requests while the migration
runs in parallel.
"""

import copy
import os
import re
import subprocess
import sys
import threading
from datetime import datetime, timezone

from config import (
    EMBEDDING_MODELS,
    MODEL_CACHE_PATH,
    POSTGRE_DATABASE_NAME,
    POSTGRE_PASSWORD,
    POSTGRE_URL,
    POSTGRE_USER,
    logger,
)

# ── migration state ───────────────────────────────────────────────────────────
_state_lock = threading.Lock()
_current: dict | None = None  # current or most-recent migration state


class MigrationAlreadyRunningError(RuntimeError):
    pass


# ── helpers ───────────────────────────────────────────────────────────────────

def _subprocess_env() -> dict:
    """
    Build the environment for the migrate.py subprocess, seeding it with the
    server's live connection config so the subprocess targets the same PostgreSQL
    instance regardless of how the server was originally started (CLI args vs
    env vars vs dotenv).
    """
    env = os.environ.copy()
    env["GENIUSAI_DATABASE_NAME"] = POSTGRE_DATABASE_NAME
    env["GENIUSAI_POSTGRES_URL"]  = POSTGRE_URL
    # Remove stale values before conditionally re-adding them.
    env.pop("GENIUSAI_POSTGRES_USER",     None)
    env.pop("GENIUSAI_POSTGRES_PASSWORD", None)
    if POSTGRE_USER:
        env["GENIUSAI_POSTGRES_USER"] = POSTGRE_USER
    if POSTGRE_PASSWORD:
        env["GENIUSAI_POSTGRES_PASSWORD"] = POSTGRE_PASSWORD
    if MODEL_CACHE_PATH:
        env["MODEL_CACHE_PATH"] = MODEL_CACHE_PATH
    return env


def _parse_progress(line: str, progress: dict) -> None:
    """
    Update *progress* in-place by extracting numbers from migrate.py log lines.

    Recognised patterns:
      "Progress — inserted: X, without-embedding: Y, failed: Z"
      "Source '…': N photos to migrate."
    """
    m = re.search(
        r"inserted:\s*(\d+).*without-embedding:\s*(\d+).*skipped:\s*(\d+).*failed:\s*(\d+)", line
    )
    if m:
        progress["inserted"]          = int(m.group(1))
        progress["without_embedding"] = int(m.group(2))
        progress["skipped"]           = int(m.group(3))
        progress["failed"]            = int(m.group(4))
        return
    m = re.search(r":\s*(\d+)\s+photos to migrate\.", line)
    if m:
        progress["total"] = int(m.group(1))


def _monitor(process: subprocess.Popen, state: dict) -> None:
    """
    Drain subprocess stdout, forward each line to the server log, update the
    shared state dict, then wait for the process to exit and mark it done.
    """
    tail: list[str] = []  # last few lines kept for error reporting

    for raw in process.stdout:
        line = raw.rstrip()
        tail.append(line)
        if len(tail) > 20:
            tail.pop(0)
        logger.info("[migrate] %s", line)
        with _state_lock:
            _parse_progress(line, state["progress"])

    process.wait()

    with _state_lock:
        state["finished_at"] = datetime.now(timezone.utc).isoformat()
        if process.returncode == 0:
            state["status"] = "completed"
        else:
            state["status"] = "failed"
            state["error"]  = "\n".join(tail[-5:]) if tail else "Process exited with code %d" % process.returncode


# ── public API ────────────────────────────────────────────────────────────────

def start_migration(target_db: str, target_model: str, batch_size: int = 32) -> dict:
    """
    Launch migrate.py as a subprocess and return the initial state snapshot.
    Raises MigrationAlreadyRunningError if a migration is already in progress.
    """
    global _current

    with _state_lock:
        if _current is not None and _current["status"] == "running":
            raise MigrationAlreadyRunningError("A migration is already in progress.")

        state: dict = {
            "status":      "running",
            "source_db":   POSTGRE_DATABASE_NAME,
            "target_db":   target_db,
            "target_model": target_model,
            "model_id":    EMBEDDING_MODELS[target_model]["model_id"],
            "started_at":  datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "error":       None,
            "progress": {
                "total":             None,
                "inserted":          0,
                "without_embedding": 0,
                "skipped":           0,
                "failed":            0,
            },
        }
        _current = state

    _migrate_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "migrate.py")
    cmd = [
        sys.executable, _migrate_script,
        "--target-db",    target_db,
        "--target-model", target_model,
        "--batch-size",   str(batch_size),
    ]

    logger.info(
        "Starting migration subprocess: '%s' → '%s' (model: %s, batch: %d)",
        POSTGRE_DATABASE_NAME, target_db, target_model, batch_size,
    )

    process = subprocess.Popen(
        cmd,
        env=_subprocess_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    thread = threading.Thread(
        target=_monitor,
        args=(process, state),
        name="migration-monitor",
        daemon=True,
    )
    thread.start()

    return copy.deepcopy(state)


def get_status() -> dict | None:
    """Return a snapshot of the current/last migration state, or None if never started."""
    with _state_lock:
        return copy.deepcopy(_current)
