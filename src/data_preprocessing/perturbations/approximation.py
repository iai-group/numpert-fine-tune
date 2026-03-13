"""Module implements numeric approximation perturbations for claims."""

import json
import random
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

PERTUBATION_METHOD = "approximation"


# =========================
# Configuration
# =========================

TARGET_LABELS = {"CARDINAL", "PERCENT", "MONEY", "QUANTITY", "TIME"}

APPROXIMATION_KEYWORDS = {
    "about",
    "around",
    "approximately",
    "roughly",
    "nearly",
    "almost",
    "close to",
    "just over",
    "just under",
    "more than",
    "less than",
}


# =========================
# Data model
# =========================


@dataclass
class NumericSpan:
    """Data class for numeric spans in text."""

    token_start: int
    token_end: int
    char_start: int
    char_end: int
    number_token_idx: int
    original_number: str
    original_value: float
    approx_true: float
    approx_false: float


# =========================
# Numeric helpers
# =========================


def exaggerate_number(n: float) -> float:
    """Exaggerates a number to create a false claim."""
    if n == 0:
        return random.choice([10.0, 50.0])

    # 1. Handle single-digit to double-digit jump
    if abs(n) < 10:
        # Pick a random double-digit "round" number
        # E.g., 10, 12, 15, 20, 25, 50
        # These are common "round" numbers people use when approximating
        # e.g., "about 20" instead of "about 18"
        val = float(random.choice([10, 12, 15, 20, 25, 50]))
        return -val if n < 0 else val

    # 2. For larger numbers, use more aggressive multipliers
    # [0.4, 0.6] creates a significant "under-count"
    # [1.6, 2.0] creates a significant "over-count"
    multiplier = random.choice([0.4, 0.5, 0.6, 1.5, 1.8, 2.0])
    raw_exaggeration = n * multiplier

    # 3. Pass it through approximate_number to keep it "clean"
    # This ensures 13,000 * 1.5 becomes 20,000 or 19,500 rather than 19,500.0
    return approximate_number(raw_exaggeration)


def approximate_number(n: float) -> float:
    """Round number to approximate value using human-preferred steps."""
    if n == 0:
        return 0

    sign = -1 if n < 0 else 1
    n = abs(n)

    # Small numbers: behaves normally (e.g., 7 -> 7)
    if n < 10:
        return sign * round(n)

    magnitude = 10 ** int(math.floor(math.log10(n)))
    normalized = n / magnitude

    # Human-preferred multipliers for "about" values
    # We include .5 steps for common midpoints (1.5, 2.5, 7.5)
    steps = [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.5, 8.0, 10.0]

    # 1. Check if the original number is already "round enough"
    # We check if it matches one of our steps exactly or at one magnitude lower
    for step in steps:
        if math.isclose(normalized, step, rel_tol=1e-7):
            return sign * n

    lower_magnitude = magnitude / 10
    if n % lower_magnitude == 0:
        lower_normalized = n / lower_magnitude
        # If it's a clean multiple of 5 or 10 at the lower level
        if lower_normalized % 5 == 0 or lower_normalized % 10 == 0:
            return sign * n

    # 2. Find the closest human-preferred step
    # For 3.2, the closest step in our list is 3.0
    # For 1.3, the closest step is 1.2 or 1.5
    best_step = min(steps, key=lambda x: abs(x - normalized))

    return sign * (best_step * magnitude)


def parse_number(token: str) -> Optional[float]:
    """Parse a numeric token into a float, handling commas."""
    try:
        return float(token.replace(",", ""))
    except ValueError:
        return None


def format_like(original: str, value: float) -> str:
    """Format the new value to match the original's formatting."""
    out = str(int(round(value)))
    if "," in original:
        return f"{int(out):,}"
    return out


# =========================
# Span discovery
# =========================


def find_numeric_token_spans(tokens, ner_tags):
    """Find spans of tokens labeled as numeric by NER tags."""
    spans = []
    i = 0
    while i < len(tokens):
        if ner_tags[i] in TARGET_LABELS:
            start = i
            while i < len(tokens) and ner_tags[i] == ner_tags[start]:
                i += 1
            spans.append((start, i))
        else:
            i += 1
    return spans


def build_token_char_map(text: str, tokens: List[str]) -> List[Tuple[int, int]]:
    """Build a mapping from token indices to character spans.

    Maps token indices to character spans in the original text.
    """
    spans = []
    pos = 0
    for tok in tokens:
        start = text.find(tok, pos)
        if start == -1:
            spans.append((-1, -1))
            continue
        end = start + len(tok)
        spans.append((start, end))
        pos = end
    return spans


def expand_span_to_include_currency(text: str, start: int) -> int:
    """Expand the character span to include preceding currency symbols."""
    i = start - 1
    while i >= 0 and text[i] == " ":
        i -= 1

    # Common currency symbols
    if i >= 0 and text[i] in "$€£₹₦":
        return i

    # Currency codes like US$, USD
    for code in ["US$", "USD", "EUR", "GBP", "INR", "NGN"]:
        if text[max(0, i - len(code) + 1) : i + 1] == code:
            return i - len(code) + 1

    return start


def extract_numeric_spans(text, tokens, ner_tags) -> List[NumericSpan]:
    """Extract numeric spans from the text based on tokens and NER tags."""
    token_char_spans = build_token_char_map(text, tokens)
    numeric_spans = []

    for t_start, t_end in find_numeric_token_spans(tokens, ner_tags):
        num_idx = None
        for i in range(t_start, t_end):
            if parse_number(tokens[i]) is not None:
                num_idx = i
                break
        if num_idx is None:
            continue

        raw_start = token_char_spans[t_start][0]
        char_start = expand_span_to_include_currency(text, raw_start)

        char_end = token_char_spans[t_end - 1][1]
        if char_start == -1 or char_end == -1:
            continue

        value = parse_number(tokens[num_idx])
        if value is None:
            continue
        approx = approximate_number(value)

        numeric_spans.append(
            NumericSpan(
                token_start=t_start,
                token_end=t_end,
                char_start=char_start,
                char_end=char_end,
                number_token_idx=num_idx,
                original_number=tokens[num_idx],
                original_value=value,
                approx_true=approx,
                approx_false=exaggerate_number(approx),
            )
        )

    return numeric_spans


# =========================
# Replacement logic
# =========================


def needs_about(prefix: str) -> bool:
    """Determine if "about" should be added based on preceding context."""
    words = prefix.lower().split()[-3:]
    return not any(w.strip(".,!?;:") in APPROXIMATION_KEYWORDS for w in words)


def is_sentence_start(prefix: str) -> bool:
    """Check if the prefix indicates the start of a sentence."""
    return not prefix.strip() or prefix.strip()[-1] in ".!?"


def apply_numeric_spans(
    text: str, spans: List[NumericSpan], is_true: bool
) -> str:
    """Apply numeric span replacements to the text."""
    for span in sorted(spans, key=lambda s: s.char_start, reverse=True):
        original = text[span.char_start : span.char_end]

        new_value = span.approx_true if is_true else span.approx_false
        formatted = format_like(span.original_number, new_value)

        # Replace only the numeric token inside the original surface span
        replaced = original.replace(span.original_number, formatted, 1)

        prefix = text[: span.char_start]
        about = ""

        if needs_about(prefix):
            about = "About " if is_sentence_start(prefix) else "about "

        text = (
            text[: span.char_start] + about + replaced + text[span.char_end :]
        )
    return text


def contains_approximation(text: str) -> bool:
    """Check if the text contains any approximation keywords."""
    return any(k in text.lower() for k in APPROXIMATION_KEYWORDS)


def perturb_claim(claim: str, tokens: List[str], ner_tags: List[str]):
    """Perturb a claim to create true and false approximation variants."""
    spans = extract_numeric_spans(claim, tokens, ner_tags)
    if not spans:
        return None, None

    true_claim = apply_numeric_spans(claim, spans, True)
    false_claim = apply_numeric_spans(claim, spans, False)

    if true_claim == claim or false_claim == claim:
        return None, None

    return true_claim, false_claim


def process_claims(input_file, output_file):
    """Process claims from input file and write perturbed claims."""
    with open(input_file, "r", encoding="utf-8") as f:
        claims = json.load(f)

    output = []
    excluded_approx = 0
    excluded_unperturbed = 0

    for claim in claims:
        text = claim["normalized_claim"]
        if contains_approximation(text):
            excluded_approx += 1
            continue

        t, f_ = perturb_claim(text, claim["tokens"], claim["ner_tags"])
        if not t:
            excluded_unperturbed += 1
            continue

        label = claim.get("label", "UNKNOWN")
        out = {
            k: v
            for k, v in claim.items()
            if k not in {"tokens", "ner_tags", "entities"}
        }
        out["pertubation_method"] = PERTUBATION_METHOD
        out[f"perturb_type1_{label}"] = t
        out[f"perturb_type2_{label}"] = f_
        output.append(out)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Print summary
    total_excluded = excluded_approx + excluded_unperturbed
    print("Processing complete:")
    print(f"  - Total input claims: {len(claims)}")
    print(f"  - Successfully perturbed: {len(output)}")
    print(f"  - Excluded (approximation keywords): {excluded_approx}")
    print(f"  - Excluded (unperturbed): {excluded_unperturbed}")
    print(f"  - Total excluded: {total_excluded}")
    print(f"\nOutput written to: {output_file}")
