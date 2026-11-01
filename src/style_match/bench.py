"""T0.1 — ARM CLIP benchmark: throughput and peak RSS, fp32 vs int8.

Run on the target (k3s-rpi20) before trusting the M2 latency budget:

    uv run python -m style_match.bench --images <dir> [--limit 50] [--int8]

Reports img/s (single image and batched), peak RSS, and decides fp32 vs int8
for the 8 GB budget (plan §2.4).
"""

import argparse
import io
import resource
import sys
import time
from pathlib import Path

import numpy as np


def _peak_rss_mb():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is bytes on macOS, KiB on Linux.
    raw = usage.ru_maxrss
    return raw / (1024 * 1024) if sys.platform == "darwin" else raw / 1024.0


def _load_images(directory, limit):
    from PIL import Image

    paths = sorted(
        path
        for path in Path(directory).iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if limit:
        paths = paths[:limit]
    images = []
    for path in paths:
        images.append(Image.open(path).convert("RGB"))
    return images


def _encode_batch(model, processor, images, device):
    import torch

    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
        return features.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="CLIP ViT-B/32 ARM benchmark")
    parser.add_argument("--images", required=True, help="Directory of test images")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--int8", action="store_true", help="Dynamic-quantize the model")
    parser.add_argument("--threads", type=int, default=None, help="torch thread count")
    args = parser.parse_args()

    import torch
    from transformers import CLIPImageProcessor, CLIPModel

    if args.threads:
        torch.set_num_threads(args.threads)
    device = "cpu"

    print("Loading openai/clip-vit-base-patch32 ...")
    load_start = time.time()
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    if args.int8:
        model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        print("Dynamic int8 quantization applied (Linear layers).")
    model.to(device).eval()
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print(f"Model loaded in {time.time() - load_start:.1f}s; peak RSS so far: {_peak_rss_mb():.0f} MB")

    images = _load_images(args.images, args.limit)
    if not images:
        print("No images found — nothing to benchmark.")
        return 2
    print(f"{len(images)} images, batch={args.batch_size}, threads={torch.get_num_threads()}")

    # Warm-up
    _encode_batch(model, processor, images[:1], device)

    single_start = time.time()
    for image in images:
        _encode_batch(model, processor, [image], device)
    single_elapsed = time.time() - single_start
    single_rate = len(images) / single_elapsed

    batch_start = time.time()
    batched = 0
    for start in range(0, len(images), args.batch_size):
        _encode_batch(model, processor, images[start : start + args.batch_size], device)
        batched += min(args.batch_size, len(images) - start)
    batch_elapsed = time.time() - batch_start
    batch_rate = batched / batch_elapsed

    probe = _encode_batch(model, processor, images[:1], device)

    print("")
    print("=== CLIP ViT-B/32 benchmark ===")
    print(f"mode            : {'int8' if args.int8 else 'fp32'}")
    print(f"single-image    : {single_rate:.2f} img/s")
    print(f"batch-{args.batch_size:<9}: {batch_rate:.2f} img/s")
    print(f"embedding dim   : {probe.shape[1]}")
    print(f"peak RSS        : {_peak_rss_mb():.0f} MB")
    print(f"10k photos      : ~{10000 / max(batch_rate, 1e-9) / 3600:.1f} h (batched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
