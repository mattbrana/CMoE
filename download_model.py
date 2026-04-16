"""One-shot Hugging Face model downloader.

Examples
--------
    # Download to the default HF cache and print the local path
    python download_model.py meta-llama/Llama-2-7b-hf

    # Download to a specific folder (handy as MODEL_PATH for run_cmoe.py)
    python download_model.py meta-llama/Llama-2-7b-hf --local-dir ./models/llama-2-7b

    # Gated repo: pass a token (or set HF_TOKEN / HUGGINGFACE_HUB_TOKEN in env)
    python download_model.py meta-llama/Llama-2-7b-hf --token hf_xxx
"""
import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face model snapshot for use with run_cmoe.py.",
    )
    parser.add_argument(
        "model",
        type=str,
        help="HF repo id (e.g. meta-llama/Llama-2-7b-hf) or local path.",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help="Directory to materialize the snapshot in. Defaults to the HF cache.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HF access token for gated/private repos. "
             "Falls back to HF_TOKEN / HUGGINGFACE_HUB_TOKEN env vars.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional git revision / branch / tag to pin.",
    )
    args = parser.parse_args()

    if os.path.isdir(args.model):
        print(f"'{args.model}' already exists locally; nothing to download.")
        print(args.model)
        return 0

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "huggingface_hub is not installed. Install with:\n"
            "    pip install huggingface_hub",
            file=sys.stderr,
        )
        return 1

    token = (
        args.token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )

    print(f"Downloading {args.model} ...")
    path = snapshot_download(
        repo_id=args.model,
        local_dir=args.local_dir,
        token=token,
        revision=args.revision,
    )
    print(f"\nModel ready at: {path}")
    print("\nUse this path as MODEL_PATH when launching run_cmoe.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
