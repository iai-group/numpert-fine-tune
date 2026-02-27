#!/usr/bin/env python3
"""Upload a finetuned model to HuggingFace Hub.

Supports merged, LoRA adapter, or checkpoint uploads. Requires HF_TOKEN set in
the environment or active huggingface-cli login.
"""

import argparse
import sys
from pathlib import Path


def check_auth():
    """Verify HF authentication is available, exit with clear message if not."""
    try:
        from huggingface_hub import whoami

        user = whoami()
        return user["name"]
    except Exception:
        print(
            "\n[ERROR] HuggingFace authentication not found.\n"
            "Please set your token via one of:\n"
            "  export HF_TOKEN=hf_...\n"
            "  huggingface-cli login\n"
        )
        sys.exit(1)


def detect_upload_type(model_path: Path) -> str:
    """Heuristically determine what kind of model model_path this is.

    Returns: 'merged', 'lora', or 'checkpoint'
    """
    files = set(f.name for f in model_path.iterdir())

    # LoRA adapter: has adapter_config.json but no full model shards
    if "adapter_config.json" in files:
        return "lora"

    # Trainer checkpoint: has optimizer/scheduler/trainer state files
    checkpoint_markers = {
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
    }
    if checkpoint_markers & files:
        return "checkpoint"

    # Merged model: has config.json + model weights
    if "config.json" in files and any(
        f.endswith((".bin", ".safetensors")) or f.startswith("model-")
        for f in files
    ):
        return "merged"

    return "unknown"


def build_repo_id(args) -> str:
    """Build repository ID from namespace and model name.

    Args:
        args: Arguments containing namespace, username, and model_name.

    Returns:
        Repository ID in format 'namespace/model_name'.
    """
    namespace = args.namespace or args.username
    return f"{namespace}/{args.model_name}"


def upload(args):
    """Upload model to HuggingFace Hub.

    Detects model type, creates repository, and uploads model files.
    Handles LoRA adapters, merged models, and checkpoints.

    Args:
        args: Command-line arguments with model path and upload options.
    """
    from huggingface_hub import HfApi

    model_path = Path(args.model_path).resolve()
    if not model_path.exists():
        print(f"[ERROR] model_path not found: {model_path}")
        sys.exit(1)

    # Auto-detect type unless overridden
    upload_type = args.type or detect_upload_type(model_path)
    if upload_type == "unknown":
        print(
            "[WARN] Could not auto-detect model_path type "
            "(merged/lora/checkpoint).\n"
            "       Pass --type explicitly if the upload looks wrong."
        )
    else:
        print(f"[✓] Upload type detected: {upload_type}")

    repo_id = build_repo_id(args)
    private = not args.public
    visibility = "public" if args.public else "private"
    print(f"[✓] Visibility: {visibility}")

    api = HfApi()

    # Create repo if it doesn't exist
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True,
        )
        print("[✓] Repo ready")
    except Exception as e:
        print(f"[ERROR] Failed to create/access repo: {e}")
        sys.exit(1)

    # For checkpoints: skip optimizer/scheduler bloat unless
    # explicitly requested
    ignore_patterns = None
    if upload_type == "checkpoint" and not args.include_optimizer_states:
        ignore_patterns = [
            "optimizer.pt",
            "optimizer.bin",
            "scheduler.pt",
            "rng_state*.pth",
        ]
        print(
            "[i] Skipping optimizer/scheduler states "
            "(pass --include-optimizer-states to keep them)."
        )

    # Upload
    display_path = (
        Path(*model_path.parts[-3:])
        if len(model_path.parts) >= 3
        else model_path
    )
    print(f"\n[→] Uploading from: .../{display_path}")
    print("    This may take a while for large models...\n")

    try:
        api.upload_folder(
            folder_path=str(model_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message=args.commit_message,
            ignore_patterns=ignore_patterns,
        )
    except Exception as e:
        print(f"[ERROR] Upload failed: {e}")
        sys.exit(1)

    print("\n[✓] Upload complete!\n")

    # Contextual reminders based on upload type
    if upload_type == "lora":
        print(
            "[i] Note: You uploaded LoRA adapter weights.\n"
            "    Users will need the base model + 'peft' to load this.\n"
            "    Consider also uploading a merged model for broader "
            "usability.\n"
        )
    elif upload_type == "checkpoint":
        print(
            "[i] Note: You uploaded a training checkpoint.\n"
            "    This is useful for resuming training but is NOT "
            "directly\n"
            "    loadable as a standard inference model.\n"
        )


def main():
    """Parse arguments and execute model upload."""
    parser = argparse.ArgumentParser(
        description="Upload a finetuned model to HuggingFace Hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload a merged model (auto-detects type)
  python upload_to_hf.py ./merged_model my-llama-finetune

  # Upload LoRA adapter weights only
  python upload_to_hf.py ./lora_weights my-llama-lora --type lora

  # Upload to an org namespace, public
  python upload_to_hf.py ./merged_model my-model --namespace my-org --public

  # Upload a specific checkpoint, keeping optimizer states
  python upload_to_hf.py ./checkpoint-1000 my-model-ckpt1000
  --type checkpoint
  --include-optimizer-states
        """,
    )

    parser.add_argument(
        "--model_path",
        default=None,
        help="Path to the model model_path (merged model, LoRA weights, "
        "or checkpoint).",
    )
    parser.add_argument(
        "--model_name",
        default=None,
        help="Name of the model repo on HuggingFace "
        "(e.g. 'my-llama-finetune').",
    )
    parser.add_argument(
        "--namespace",
        default=None,
        help="Org or username namespace to upload under. "
        "Defaults to your HF username.",
    )
    parser.add_argument(
        "--type",
        choices=["merged", "lora", "checkpoint"],
        default=None,
        help="Explicitly set the upload type. Auto-detected if not provided.",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        default=False,
        help="Make the repo public. Defaults to private.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload finetuned model via upload_to_hf.py",
        help="Commit message for the upload.",
    )
    parser.add_argument(
        "--include-optimizer-states",
        action="store_true",
        default=False,
        help="For checkpoints: include optimizer/scheduler state files "
        "(large, only needed for resuming training).",
    )

    args = parser.parse_args()

    # Auth check — populates username for namespace fallback
    print()
    args.username = check_auth()
    upload(args)


if __name__ == "__main__":
    main()
