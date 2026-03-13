"""Run naming utilities for training and inference outputs.

Implements a human-readable naming convention that combines descriptive words
(adjectives, nouns, verbs) with model names to create unique identifiers for
training and inference runs.
"""

import random
from datetime import datetime
from pathlib import Path
from typing import Optional

# Word lists for name generation
ADJECTIVES = [
    "ancient",
    "bright",
    "calm",
    "daring",
    "eager",
    "fierce",
    "gentle",
    "happy",
    "icy",
    "jolly",
    "keen",
    "lazy",
    "mighty",
    "noble",
    "proud",
    "quick",
    "rare",
    "sharp",
    "tiny",
    "vivid",
    "wild",
    "young",
    "zealous",
    "bold",
    "clever",
    "dark",
    "fast",
    "grand",
    "hard",
    "loud",
    "rich",
    "silent",
    "swift",
    "warm",
    "wise",
]
# De-duplicate to avoid biased sampling
ADJECTIVES = list(dict.fromkeys(ADJECTIVES))

NOUNS = [
    "ape",
    "bear",
    "cat",
    "deer",
    "eagle",
    "fox",
    "giraffe",
    "horse",
    "ibis",
    "jaguar",
    "kite",
    "leopard",
    "monkey",
    "nightingale",
    "owl",
    "panda",
    "quail",
    "raven",
    "seal",
    "tiger",
    "unicorn",
    "vulture",
    "walrus",
    "xenops",
    "yak",
    "zebra",
    "ant",
    "badger",
    "camel",
    "dolphin",
    "emu",
    "falcon",
    "gorilla",
    "hawk",
    "impala",
    "koala",
    "lion",
    "moose",
    "narwhal",
    "otter",
    "penguin",
    "rabbit",
    "squirrel",
    "turtle",
    "viper",
    "wolf",
    "xerus",
    "yellowhammer",
    "zebu",
]

VERBS = [
    "arranges",
    "builds",
    "calculates",
    "designs",
    "evolves",
    "flows",
    "grows",
    "helps",
    "innovates",
    "jumps",
    "kindles",
    "learns",
    "manifests",
    "navigates",
    "optimizes",
    "processes",
    "questions",
    "refines",
    "solves",
    "transforms",
    "updates",
    "validates",
    "witnesses",
    "executes",
    "yields",
    "zeroes",
]


def generate_run_name(
    model_name: str,
    timestamp: Optional[datetime] = None,
    perturb_type_names: str = "",
    base_path: Optional[str] = None,
    max_attempts: int = 1000,
) -> str:
    """Generate a unique, human-readable run name for a training/inference.

    The name combines a random adjective, noun, and verb with the model name,
    creating descriptive identifiers like "Qwen3-8B-mighty-lion-learns". If a
    directory with that name already exists at the specified base_path,
    generates new combinations until finding a unique name.
    If all combinations are exhausted, appends a numeric suffix.

    Args:
        model_name: Short or full model name (e.g. ``"Qwen3-8B"``).
        timestamp: Optional datetime parameter (kept for backwards
            compatibility, not used in name generation).
        perturb_type_names: Optional perturbation descriptor appended to
            the name (used by training runs).
        base_path: Base directory path to check for existing run names. If
            provided, the function will check whether directories with generated
            names already exist at this path. If None, returns a name that is
            statistically unique but not checked against the filesystem.
        max_attempts: Maximum number of name combinations to try before
            appending a numeric suffix (default 1000).

    Returns:
        A unique run name string in the format:
        "{model_name}-{adjective}-{noun}-{verb}" or with numeric
        suffix if needed.
        Includes perturb_type_names as a suffix if provided.
    """
    base_path_obj = Path(base_path) if base_path else None

    # Sanitize model_name: replace path separators to avoid nested dirs
    safe_model_name = model_name.replace("/", "-")

    used_names = set()
    if base_path_obj and base_path_obj.is_dir():
        try:
            used_names = {d.name for d in base_path_obj.iterdir() if d.is_dir()}
        except OSError:
            pass

    # Try to generate a unique combination
    for attempt in range(max_attempts):
        adjective = random.choice(ADJECTIVES)
        noun = random.choice(NOUNS)
        verb = random.choice(VERBS)
        name = f"{safe_model_name}-{adjective}-{noun}-{verb}"

        if perturb_type_names:
            name = f"{name}_{perturb_type_names}"

        if not used_names or name not in used_names:
            return name

    # If all combinations exhausted, append numeric suffix
    base_name = (
        f"{safe_model_name}-{random.choice(ADJECTIVES)}-"
        f"{random.choice(NOUNS)}-{random.choice(VERBS)}"
    )
    if perturb_type_names:
        base_name = f"{base_name}_{perturb_type_names}"

    # Find first available numbered suffix
    counter = 1
    while f"{base_name}_v{counter}" in used_names:
        counter += 1

    return f"{base_name}_v{counter}"


def build_output_dir_name(
    model_name: str,
    rank: int,
    start_epoch: int,
    load_in_4bit: bool = True,
) -> str:
    """Build the subdirectory name for prediction / model outputs.

    Encapsulates the ``rank_postfix`` and ``quant_status`` logic that was
    previously duplicated across inference and training modules.

    Args:
        model_name: Model identifier.
        rank: LoRA rank.  Rank 16 produces no suffix.
        start_epoch: Epoch tag included in the directory name.
        load_in_4bit: Whether 4-bit quantisation is used (affects suffix).

    Returns:
        A directory *name* (not full path) such as
        ``"Qwen3-8B_r32-epoch-1"``.
    """
    # Sanitize model_name: replace path separators to avoid nested dirs
    safe_model_name = model_name.replace("/", "-").replace("\\", "-")
    rank_postfix = "" if rank == 16 else f"_r{rank}"
    quant_status = "" if load_in_4bit else "-unquantized"
    return f"{safe_model_name}{rank_postfix}{quant_status}-epoch-{start_epoch}"
