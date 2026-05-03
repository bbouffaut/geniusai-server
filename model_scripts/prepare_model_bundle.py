import argparse
import os

from huggingface_hub import snapshot_download


def prepare_model_bundle(model_id: str, output_dir: str):
    """
    Prepares a Hugging Face text embedding model for bundling with the application.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Downloading model {model_id} from Hugging Face into {output_dir}...")
    path = snapshot_download(repo_id=model_id, local_dir=output_dir)
    print(f"Bundled model directory ready at: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare text embedding model bundle for LrGenius AI Server.")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-Embedding-0.6B", help="Hugging Face model ID.")
    parser.add_argument("--output_dir", type=str, default="dist/models", help="The output directory for the bundled model files.")
    args = parser.parse_args()

    prepare_model_bundle(args.model_id, args.output_dir)
