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

    Returns: 'merged', 'lora', 'checkpoint', or 'gguf'
    """
    # Single GGUF file
    if model_path.is_file() and model_path.suffix == ".gguf":
        return "gguf"

    if not model_path.is_dir():
        return "unknown"

    files = set(f.name for f in model_path.iterdir())

    # Directory containing GGUF files
    if any(f.endswith(".gguf") for f in files):
        return "gguf"

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


def _build_ignore_patterns(upload_type: str, include_optimizer_states: bool):
    """Return ignore patterns for folder uploads, or None if not applicable."""
    if upload_type == "checkpoint" and not include_optimizer_states:
        print(
            "[i] Skipping optimizer/scheduler states "
            "(pass --include-optimizer-states to keep them)."
        )
        return [
            "optimizer.pt",
            "optimizer.bin",
            "scheduler.pt",
            "rng_state*.pth",
        ]
    return None


def _validate_model_path_shape(model_path: Path, upload_type: str):
    """Validate model_path shape and file extension against upload_type."""
    # merged / lora / checkpoint must be directories; only gguf may be a file.
    if upload_type in {"merged", "lora", "checkpoint"} and model_path.is_file():
        print(
            f"[ERROR] --type {upload_type} requires a directory, "
            f"but got a file: {model_path}\n"
            "        Pass --type gguf if you intended to upload a GGUF file."
        )
        sys.exit(1)

    if model_path.is_file():
        if upload_type != "gguf":
            print(
                "[ERROR] Single-file upload is only supported for GGUF files.\n"
                f"        Detected type: {upload_type}\n"
                f"        File: {model_path}\n"
                "        Pass --type gguf and provide a .gguf file."
            )
            sys.exit(1)

        if model_path.suffix.lower() != ".gguf":
            print(
                "[ERROR] --type gguf requires a .gguf file when model_path "
                "is a file.\n"
                f"        Got: {model_path}"
            )
            sys.exit(1)

    if upload_type == "gguf" and model_path.is_dir():
        if not any(p.suffix.lower() == ".gguf" for p in model_path.iterdir()):
            print(
                "[ERROR] --type gguf requires at least one .gguf file in "
                f"directory: {model_path}"
            )
            sys.exit(1)


def _do_upload(
    api, model_path: Path, repo_id: str, commit_message: str, ignore_patterns
):
    """Upload a single file or folder to the Hub, exiting on failure."""
    try:
        if model_path.is_file():
            api.upload_file(
                path_or_fileobj=str(model_path),
                path_in_repo=model_path.name,
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message,
            )
        else:
            api.upload_folder(
                folder_path=str(model_path),
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message,
                ignore_patterns=ignore_patterns,
            )
    except Exception as e:
        print(f"[ERROR] Upload failed: {e}")
        sys.exit(1)


def _print_type_reminder(upload_type: str):
    """Print a contextual post-upload reminder based on upload type."""
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
    elif upload_type == "gguf":
        print(
            "[i] Note: You uploaded GGUF quantized model file(s).\n"
            "    These can be used directly with llama.cpp, ollama, "
            "and other\n"
            "    GGUF-compatible inference engines.\n"
        )


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

    _validate_model_path_shape(model_path, upload_type)

    api = HfApi()

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

    ignore_patterns = _build_ignore_patterns(
        upload_type, args.include_optimizer_states
    )

    display_path = (
        Path(*model_path.parts[-3:])
        if len(model_path.parts) >= 3
        else model_path
    )
    print(f"\n[→] Uploading from: .../{display_path}")
    print("    This may take a while for large models...\n")

    _do_upload(api, model_path, repo_id, args.commit_message, ignore_patterns)

    print("\n[✓] Upload complete!\n")
    _print_type_reminder(upload_type)


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
        choices=["merged", "lora", "checkpoint", "gguf"],
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
