import os
import time
import signal
from config import DB_PATH, logger, IMAGE_MODEL_ID, CLIP_MODEL_NAME, TORCH_DEVICE
import sys
import open_clip
import threading
import datetime
import gc
import torch
from tqdm import tqdm
from huggingface_hub import HfApi, snapshot_download, hf_hub_download

# Lazy-loadable global model instances
# model, processor and tokenizer start as None and will be loaded on first use.
model = None
processor = None  # This will hold the image preprocessor
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


class DownloadProgressTracker(tqdm):
    bytes_downloaded = 0
    total_size = 1  # Avoid division by zero, will be updated

    def __init__(self, *args, **kwargs):
        kwargs.pop('name', None)
        super().__init__(*args, **kwargs, file=open(os.devnull, 'w'))
        # We only care about byte-level progress

        with _download_lock:
            # Set the total for the status, this might be called multiple times
            # but it is fine. The final value in _download_status will be from the
            # _download_clip_model_thread function.
            _download_status["total"] = DownloadProgressTracker.total_size
            _download_status["unit"] = "B"

    def update(self, n=1):
        super().update(n)
        with _download_lock:
            DownloadProgressTracker.bytes_downloaded += n
            _download_status["progress"] = DownloadProgressTracker.bytes_downloaded


def get_download_status():
    with _download_lock:
        return _download_status

_download_thread = None
_download_lock = threading.Lock()


def _download_clip_model_thread():
    global _download_status
    with _download_lock:
        if _download_status["status"] == "downloading":
            logger.warning("Download already in progress.")
            return

    logger.info("Starting CLIP model download in background thread.")
    try:
        api = HfApi()
        model_info = api.model_info(IMAGE_MODEL_ID, files_metadata=True)
        total_size = sum(f.size for f in model_info.siblings if f.size)

        DownloadProgressTracker.total_size = total_size
        DownloadProgressTracker.bytes_downloaded = 0

        with _download_lock:
            _download_status = {
                "status": "downloading",
                "progress": 0,
                "total": total_size,
                "error": None,
                "unit": "B"
            }

        path = snapshot_download(
            repo_id=IMAGE_MODEL_ID,
            tqdm_class=DownloadProgressTracker
        )

        with _download_lock:
            _download_status["status"] = "completed"
        logger.info(f"CLIP model downloaded to {path}")
    except Exception as e:
        logger.error(f"Error downloading CLIP model in background: {e}", exc_info=True)
        with _download_lock:
            _download_status["status"] = "error"
            _download_status["error"] = str(e)

def start_download_clip_model():
    global _download_thread
    with _download_lock:
        if _download_thread and _download_thread.is_alive():
            logger.warning("Download thread is already running.")
            return
        _download_thread = threading.Thread(target=_download_clip_model_thread)
        _download_thread.daemon = True
        _download_thread.start()


def _set_last_used():
    global _last_used
    _last_used = datetime.datetime.utcnow()


def _needs_unload():
    if _last_used is None:
        return False
    delta = datetime.datetime.utcnow() - _last_used
    return delta.total_seconds() >= IDLE_UNLOAD_SECONDS


def load_model():
    """Load the OpenCLIP model (idempotent)."""
    global model, processor, tokenizer
    with _model_lock:
        if model is not None:
            _set_last_used()
            return

        try:
            logger.info("Trying to load open_clip model from local cache")

            try:
                cached_model_file = hf_hub_download(
                    repo_id=IMAGE_MODEL_ID,
                    filename="open_clip_model.safetensors",
                    local_files_only=True
                )

                cached_model_dir = os.path.dirname(cached_model_file)

                logger.info(f"Checking for cached model at: {cached_model_dir}")
                
                # Check if local model directory exists (production/bundled scenario)
                if os.path.isdir(cached_model_dir):
                    # Verify model files exist
                    config_file = os.path.join(cached_model_dir, 'open_clip_config.json')
                    weights_file = os.path.join(cached_model_dir, 'open_clip_model.safetensors')

                    if os.path.isfile(config_file) and os.path.isfile(weights_file):
                        local_model_uri = f"local-dir:{cached_model_dir}"
                        logger.info(f"Loading OpenCLIP model from bundled directory: {cached_model_dir}")
                        model_obj, _, proc = open_clip.create_model_and_transforms(
                            local_model_uri,
                            pretrained=None
                        )
                        tok = open_clip.get_tokenizer(local_model_uri)
                        _set_last_used()
                        logger.info("Loaded OpenCLIP model (lazy)")
                    else:
                        logger.warning(f"Bundled model directory exists but required files missing")
                        logger.warning(f"Config file exists: {os.path.isfile(config_file)}")
                        logger.warning(f"Weights file exists: {os.path.isfile(weights_file)}")
                        raise FileNotFoundError("Bundled model files incomplete")

                    try:
                        model_obj.to(TORCH_DEVICE)
                        logger.info(f"Text and vision model moved to {TORCH_DEVICE}")
                    except Exception as e:
                        logger.warning(f"Failed to move text and vision model to {TORCH_DEVICE}: {e}.")

                    model = model_obj
                    processor = proc
                    tokenizer = tok

            except Exception as e:
                logger.warning("OpenCLIP model not found in local cache. Please download the model first, if you want to use it.")

        except Exception as e:
            logger.error(f"Failed to load OpenCLIP model (lazy): {e}", exc_info=True)
            raise



def unload_model():
    """Unload the loaded model to free GPU/CPU memory."""
    global model, processor, tokenizer
    with _model_lock:
        if model is None and processor is None and tokenizer is None:
            return

        logger.info("Unloading OpenCLIP model due to inactivity...")
        try:
            # If the model is a torch module, try moving it to cpu and delete the reference.
            try:
                if hasattr(model, "to"):
                    model.to("cpu")
            except Exception:
                pass

            model = None
            processor = None
            tokenizer = None

            # Best-effort free memory for CUDA
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            # Force a GC pass
            gc.collect()
            logger.info("Unloaded OpenCLIP model.")
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


def get_model():
    """Return the model, loading it lazily if needed."""
    load_model()
    _ensure_unloader_thread()
    return model


def get_processor():
    load_model()
    _ensure_unloader_thread()
    return processor


def get_tokenizer():
    load_model()
    _ensure_unloader_thread()
    return tokenizer

def get_db_dir():
    return os.path.dirname(DB_PATH)

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
