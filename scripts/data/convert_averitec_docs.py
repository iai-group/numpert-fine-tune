"""Convert AVeriTeC dev Q&A evidence to abstractive passages in-place."""

import json
import time
from pathlib import Path

import litellm

DATA_ROOT = Path(__file__).parent.parent.parent / "data" / "averitec"

DEV_NUMERICAL = DATA_ROOT / "averitec_dev_numerical.json"
TRAIN_NUMERICAL = DATA_ROOT / "averitec_train_numerical.json"
PERTURBED_DEV = DATA_ROOT / "perturbed" / "dev_split"
PERTURBED_TRAIN = DATA_ROOT / "perturbed" / "train_split"

# Same model as before (Claude Haiku 4.5), served via OpenRouter.
# Requires the OPENROUTER_API_KEY environment variable.
MODEL = "openrouter/anthropic/claude-haiku-4.5"

SYSTEM = (
    "You convert retrieved question-answer evidence into a single coherent abstractive passage. "
    "Preserve all facts, numbers, and named entities exactly. "
    "Output only the passage — no preamble, no explanation, no bullet points."
)

PROMPT = """\
Convert the following Q&A evidence into a single fluent, abstractive passage that preserves every factual detail:

{qa}

Output only the passage."""

# LABELS_TO_PROCESS = ["Conflicting", "True", "False"]
LABELS_TO_PROCESS = ["Conflicting"]


def convert_doc(qa_text: str) -> str:
    """Converts Q&A evidence into an abstractive passage via OpenRouter.

    Args:
        qa_text: The question-answer evidence text to convert.

    Returns:
        The abstractive passage produced by the model.
    """
    response = litellm.completion(
        model=MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": PROMPT.format(qa=qa_text)},
        ],
    )
    return response.choices[0].message.content.strip()


def main() -> None:
    """Converts all dev docs and patches dev + perturbed files in-place."""
    # Build claim_id → original doc mapping from dev numerical file
    process_data("dev")
    # process_data("train")


def process_data(split):
    file = None
    if split == "dev":
        file = DEV_NUMERICAL
    elif split == "train":
        file = TRAIN_NUMERICAL
    if file is None or not file.exists():
        print(f"File not found for split '{split}': {file}")
        return
    with open(file) as f:
        dev_records = json.load(f)

    id_to_doc: dict[str, str] = {
        r["claim_id"]: r["doc"]
        for r in dev_records
        if r["label"] in LABELS_TO_PROCESS
    }
    print(f"Converting {len(id_to_doc)} unique docs...")

    id_to_new_doc: dict[str, str] = {}
    for i, (claim_id, qa_doc) in enumerate(id_to_doc.items(), 1):
        print(f"  [{i}/{len(id_to_doc)}] {claim_id}", end="", flush=True)
        new_doc = convert_doc(qa_doc)
        id_to_new_doc[claim_id] = new_doc
        print(f" — {len(new_doc)} chars")
        # Small delay to stay within rate limits
        time.sleep(0.1)

    # Patch dev numerical file
    for r in dev_records:
        if r["label"] in LABELS_TO_PROCESS:
            r["doc"] = id_to_new_doc[r["claim_id"]]
    with open(DEV_NUMERICAL, "w") as f:
        json.dump(dev_records, f, indent=2, ensure_ascii=False)
    print(f"Updated {DEV_NUMERICAL}")

    # Patch all perturbed dev_split files
    for perturbed_file in sorted(PERTURBED_DEV.glob("*.json")):
        with open(perturbed_file) as f:
            records = json.load(f)
        updated = 0
        for r in records:
            cid = r.get("claim_id", "")
            if cid in id_to_new_doc and r.get("label", "") in LABELS_TO_PROCESS:
                r["doc"] = id_to_new_doc[cid]
                updated += 1
        with open(perturbed_file, "w") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"Updated {perturbed_file.name} ({updated} records)")

    print("Done.")


if __name__ == "__main__":
    main()
