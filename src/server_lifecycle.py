import os
import time
import signal
from config import (
    DATA_DIR,
    FETCH_MODELS,
    MODEL_CACHE_PATH,
    TEXT_EMBEDDING_MAX_LENGTH,
    TEXT_EMBEDDING_MODEL_ID,
    TEXT_EMBEDDING_QUERY_INSTRUCTION,
    TORCH_DEVICE,
    logger,
)
import threading
import datetime
import gc
import torch
import torch.nn.functional as F
import tqdm
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import snapshot_download, HfApi


# Lazy-loadable global model instances
# model and tokenizer start as None and will be loaded on first use.
model = None
tokenizer = None

# Idle-unload handling
# If the model hasn't been used for this many seconds, it will be unloaded to free memory.
IDLE_UNLOAD_SECONDS = 30 * 60  # 30 minutes
_last_used = None
_model_lock = threading.RLock()
_unloader_thread = None

_download_status = {
    "status": "idle",
    "progress": 0,
    "total": 0,
    "error": None
}


def _embedding_model_snapshot_download(**kwargs):
    if MODEL_CACHE_PATH:
        logger.info(f"Using embedding model cache path: {MODEL_CACHE_PATH}")
        kwargs["cache_dir"] = MODEL_CACHE_PATH
    return snapshot_download(repo_id=TEXT_EMBEDDING_MODEL_ID, **kwargs)


class DownloadProgressTracker(tqdm.tqdm):
    # Statische Zähler für den gesamten Prozess
    global_bytes_downloaded = 0
    global_total_size = 0
    
    def __init__(self, *args, **kwargs):
        # 1. Mute: Die Ausgabe geht ins Nichts (devnull)
        kwargs['file'] = open(os.devnull, 'w')
        kwargs.pop('name', None)
        
        # Original init aufrufen
        super().__init__(*args, **kwargs)

    def update(self, n=1):
        # Interne Logik von tqdm weiterlaufen lassen (für Timings etc.)
        super().update(n)
        
        # 2. Daten abgreifen
        # Wir filtern auf Bytes, um nicht Dateianzahl-Balken mitzuzählen
        if n > 0 and self.unit in ['B', 'b', None]:
             with _counter_lock:
                DownloadProgressTracker.global_bytes_downloaded += n
                
                # GUI Status Update
                with _download_lock:
                    current = DownloadProgressTracker.global_bytes_downloaded
                    total = DownloadProgressTracker.global_total_size
                    
                    # Cap auf 100% (falls tqdm etwas überschießt)
                    if total > 0 and current > total:
                        current = total
                    
                    _download_status["progress"] = current


def get_download_status():
    with _download_lock:
        return _download_status

_download_thread = None
_download_lock = threading.Lock()
_counter_lock = threading.Lock()


def _download_embedding_model_thread():
    global _download_status
    with _download_lock:
        if _download_status["status"] == "downloading":
            logger.warning("Download already in progress.")
            return

    logger.info("Starting text embedding model download in background thread.")

    try:
        api = HfApi()
        model_info = api.model_info(TEXT_EMBEDDING_MODEL_ID, files_metadata=True)
        total_size = sum(f.size for f in model_info.siblings if f.size)

        DownloadProgressTracker.global_total_size = total_size
        DownloadProgressTracker.global_bytes_downloaded = 0

        with _download_lock:
            _download_status = {
                "status": "downloading",
                "progress": 0,
                "total": total_size,
                "error": None,
                "unit": "B"
            }

        path = _embedding_model_snapshot_download(
            # tqdm_class=DownloadProgressTracker
        )

        with _download_lock:
            _download_status["status"] = "completed"
        logger.info(f"Text embedding model downloaded to {path}")
    except Exception as e:
        logger.error(f"Error downloading text embedding model in background: {e}", exc_info=True)
        with _download_lock:
            _download_status["status"] = "error"
            _download_status["error"] = str(e)


def start_download_embedding_model():
    global _download_thread
    with _download_lock:
        if _download_thread and _download_thread.is_alive():
            logger.warning("Download thread is already running.")
            return
        _download_thread = threading.Thread(target=_download_embedding_model_thread)
        _download_thread.daemon = True
        _download_thread.start()


def start_download_clip_model():
    """Backward-compatible name for older clients/routes."""
    start_download_embedding_model()


def _set_last_used():
    global _last_used
    _last_used = datetime.datetime.utcnow()


def _needs_unload():
    if _last_used is None:
        return False
    delta = datetime.datetime.utcnow() - _last_used
    return delta.total_seconds() >= IDLE_UNLOAD_SECONDS


def load_model():
    """Load the Hugging Face text embedding model (idempotent)."""
    global model, tokenizer
    with _model_lock:
        if model is not None:
            _set_last_used()
            return

        try:
            logger.info(f"Trying to load text embedding model from cache: {TEXT_EMBEDDING_MODEL_ID}")

            try:
                cached_model_dir = _embedding_model_snapshot_download(
                    local_files_only=not FETCH_MODELS,
                )

                logger.info(f"Checking for cached model at: {cached_model_dir}")

                if os.path.isdir(cached_model_dir):
                    logger.info(f"Loading text embedding model from directory: {cached_model_dir}")
                    tok = AutoTokenizer.from_pretrained(cached_model_dir, padding_side="left")
                    model_dtype = torch.float16 if TORCH_DEVICE in ("cuda", "mps") else torch.float32
                    model_obj = AutoModel.from_pretrained(cached_model_dir, torch_dtype=model_dtype)

                    try:
                        model_obj.to(TORCH_DEVICE)
                        logger.info(f"Text embedding model moved to {TORCH_DEVICE}")
                    except Exception as e:
                        logger.warning(f"Failed to move text embedding model to {TORCH_DEVICE}: {e}.")

                    model_obj.eval()
                    model = model_obj
                    tokenizer = tok
                    _set_last_used()
                    logger.info("Loaded text embedding model (lazy)")
                else:
                    raise FileNotFoundError("Text embedding model cache directory not found")

            except Exception as e:
                if FETCH_MODELS:
                    logger.warning(f"Failed to load text embedding model from cache or download. This can happen if the model has not fully downloaded, is corrupted, or if there is a configuration issue. The error was: {e}", exc_info=True)
                else:
                    logger.warning(
                        "Failed to load text embedding model from local cache. If the model has not been downloaded yet, "
                        "start the server with --fetch-models to enable downloading. "
                        f"The error was: {e}",
                        exc_info=True
                    )

        except Exception as e:
            logger.error(f"Failed to load text embedding model (lazy): {e}", exc_info=True)
            raise



def unload_model():
    """Unload the loaded model to free GPU/CPU memory."""
    global model, tokenizer
    with _model_lock:
        if model is None and tokenizer is None:
            return

        logger.info("Unloading text embedding model due to inactivity...")
        try:
            # If the model is a torch module, try moving it to cpu and delete the reference.
            try:
                if hasattr(model, "to"):
                    model.to("cpu")
            except Exception:
                pass

            model = None
            tokenizer = None

            # Best-effort free memory for CUDA
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            # Force a GC pass
            gc.collect()
            logger.info("Unloaded text embedding model.")
        except Exception as e:
            logger.warning(f"Error while unloading model: {e}")


def _idle_unloader_loop():
    """Background thread which periodically checks whether the model should be unloaded."""
    global _unloader_thread
    logger.info("Starting server_lifecycle idle unloader thread")
    try:
        while True:
            time.sleep(60)
            try:
                if _needs_unload():
                    unload_model()
            except Exception:
                logger.exception("Error checking/unloading model in background thread")
    finally:
        logger.info("Server_lifecycle idle unloader thread exiting")


def _ensure_unloader_thread():
    global _unloader_thread
    if _unloader_thread is None or not _unloader_thread.is_alive():
        _unloader_thread = threading.Thread(target=_idle_unloader_loop, daemon=True, name="server_lifecycle_unloader")
        _unloader_thread.start()


def _last_token_pool(last_hidden_states, attention_mask):
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]

    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def _format_embedding_input(text, input_type):
    if input_type == "query" and not text.startswith("Instruct:"):
        return f"Instruct: {TEXT_EMBEDDING_QUERY_INSTRUCTION}\nQuery:{text}"
    return text


def embed_texts(texts, input_type="document"):
    """Encode texts into normalized metadata-search embeddings."""
    if isinstance(texts, str):
        texts = [texts]

    if not texts:
        return []

    texts = [str(text) for text in texts if text is not None]
    if not texts:
        return []

    load_model()
    _ensure_unloader_thread()
    if model is None or tokenizer is None:
        logger.error("Text embedding model not initialized.")
        return None

    input_texts = [_format_embedding_input(text, input_type) for text in texts]

    try:
        batch = tokenizer(
            input_texts,
            max_length=TEXT_EMBEDDING_MAX_LENGTH,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(TORCH_DEVICE)

        with torch.no_grad():
            outputs = model(**batch)
            embeddings = _last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
            normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
            _set_last_used()
            return normalized_embeddings.cpu().numpy()
    except Exception as e:
        logger.error(f"Failed to generate text embeddings: {e}", exc_info=True)
        return None


def embed_query(text):
    embeddings = embed_texts([text], input_type="query")
    if embeddings is None or len(embeddings) == 0:
        return None
    return embeddings[0]


def embed_document(text):
    embeddings = embed_texts([text], input_type="document")
    if embeddings is None or len(embeddings) == 0:
        return None
    return embeddings[0]


def get_model():
    """Return the model, loading it lazily if needed."""
    load_model()
    _ensure_unloader_thread()
    return model


def get_processor():
    """Backward-compatible no-op; metadata embeddings do not use an image processor."""
    return None


def get_tokenizer():
    load_model()
    _ensure_unloader_thread()
    return tokenizer

def get_db_dir():
    return DATA_DIR

def write_pid_file():
    pid_file = os.path.join(get_db_dir(), "lrgenius-server.pid")
    with open(pid_file, "w") as f:
        f.write(str(os.getpid()))

def remove_pid_file():
    pid_file = os.path.join(get_db_dir(), "lrgenius-server.pid")
    try:
        os.remove(pid_file)
    except FileNotFoundError:
        pass

def write_ok_file():
    ok_file = os.path.join(get_db_dir(), "lrgenius-server.OK")
    with open(ok_file, "w") as f:
        f.write("OK\n")

def remove_ok_file():
    ok_file = os.path.join(get_db_dir(), "lrgenius-server.OK")
    try:
        os.remove(ok_file)
    except FileNotFoundError:
        pass

def request_shutdown():
    logger.info("Shutdown request received")
    time.sleep(1) # Give time for the response to be sent
    os.kill(os.getpid(), signal.SIGINT)
