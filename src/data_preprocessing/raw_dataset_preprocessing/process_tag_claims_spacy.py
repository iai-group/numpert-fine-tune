"""Module to load, process, and save tagged claims data."""

import spacy
import json
import logging
import re
from word2number import w2n  # type: ignore
from spacy.cli import download
from spacy.language import Language
from typing import Set, Any, Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    filename="process_claims.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
)


def load_json_data(path: str) -> List[Dict[str, Any]]:
    """Load JSON data from a file and return it as a list of dictionaries."""
    with open(path, "r") as file:
        data = json.load(file)
    return data


def filter_claims(
    data: List[Dict[str, Any]], taxonomy_label: str, label: str
) -> List[Dict[str, Any]]:
    """Filter claims based on taxonomy_label and label.

    Args:
        data (List[Dict[str, Any]]): List of claim items.
        taxonomy_label (str): The taxonomy label to filter by
        (e.g., "statistical").
        label (str): The label to filter by ("True" or "False").

    Returns:
        List[Dict[str, Any]]: Filtered list of claim items.
    """
    return [
        item
        for item in data
        if item.get("taxonomy_label") == taxonomy_label
        and item.get("label") == label
    ]


def get_number_words_set() -> Set[str]:
    """Return a set of valid number words."""
    return set(
        """
        zero one two three four five six seven eight nine ten eleven twelve
        thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty
        thirty forty fifty sixty seventy eighty ninety hundred thousand
        million billion trillion point half quarter
    """.split()
    )


def get_multipliers() -> Dict[str, int]:
    """Return a dictionary of multiplier words and their values."""
    return {
        "thousand": 1000,
        "million": 1000000,
        "billion": 1000000000,
        "trillion": 1000000000000,
    }


def is_number_word(word: str) -> bool:
    """Check if a word is a valid number word."""
    word = word.lower().replace("-", "").strip(".,")
    if word in get_number_words_set():
        return True
    try:
        float(word.replace(",", ""))
        return True
    except ValueError:
        return False


def word_to_num(word: str) -> Optional[float]:
    """Convert a number word or digit string to a float."""
    word = word.lower().replace("-", "").strip(".,")
    try:
        # Try converting directly to float (for digit-based numbers)
        return float(word.replace(",", ""))
    except ValueError:
        # Try converting word-based numbers
        try:
            return float(w2n.word_to_num(word))
        except (ValueError, IndexError) as e:
            logging.debug(f"word_to_num failed for word '{word}': {e}")
            return None


def replace_number_phrases(claim: str) -> str:
    """Replace number phrases in a claim with normalized numerical values.

    Args:
        claim (str): The original claim text.

    Returns:
        str: The claim with normalized numerical values.
    """
    currency_pattern = re.compile(
        r"(?P<currency>[$£€¥])\s?"
        r"(?P<number>\d+(\.\d+)?|"
        r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
        r"eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
        r"eighty|ninety|hundred|thousand|million|billion|trillion|"
        r"point|half|quarter)+)\s?"
        r"(?P<multiplier>thousand|million|billion|trillion)\b",
        re.IGNORECASE,
    )

    # Pattern 2: Number (digits or words) and multiplier without currency symbol
    number_pattern = re.compile(
        r"(?P<number>\d+(\.\d+)?|"
        r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
        r"eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
        r"eighty|ninety|hundred|thousand|million|billion|trillion|"
        r"point|half|quarter)+)\s?"
        r"(?P<multiplier>thousand|million|billion|trillion)\b",
        re.IGNORECASE,
    )

    def replace_match(m):
        # Check if 'currency' group exists
        currency = (
            m.group("currency")
            if "currency" in m.groupdict() and m.group("currency")
            else ""
        )
        number = m.group("number")
        multiplier = m.group("multiplier").lower()

        logging.debug(
            f"Matched phrase: "
            f"Currency='{currency}', "
            f"Number='{number}', "
            f"Multiplier='{multiplier}'"
        )

        # Convert number to float
        num = word_to_num(number)
        if num is None:
            logging.debug(f"Failed to convert number '{number}' to float.")
            return m.group(0)  # Return the original phrase if conversion fails

        # Get multiplier value
        multiplier_value = get_multipliers().get(multiplier, 1)
        if multiplier_value is None:
            logging.debug(f"Unknown multiplier '{multiplier}'.")
            return m.group(
                0
            )  # Return the original phrase if multiplier is unknown

        # Calculate final number
        final_num = num * multiplier_value
        logging.debug(f"Calculated final_num: {final_num}")

        # Format the number with commas and no decimal places
        formatted_num = f"{final_num:,.0f}"
        logging.debug(f"Formatted number: {formatted_num}")

        # Prepend the currency symbol if present
        return f"{currency}{formatted_num}" if currency else formatted_num

    # First replace currency patterns
    normalized_claim = currency_pattern.sub(replace_match, claim)
    logging.debug(f"After currency replacement: '{normalized_claim}'")

    # Then replace number patterns without currency symbols
    normalized_claim = number_pattern.sub(replace_match, normalized_claim)
    logging.debug(f"After number replacement: '{normalized_claim}'")

    return normalized_claim


def process_claim(claim: str) -> str:
    """Processes claim by converting number phrases into numerical digits.

    Args:
        claim (str): The original claim text.

    Returns:
        str: The claim with normalized numerical values.
    """
    normalized_claim = replace_number_phrases(claim)
    logging.debug(f"Original claim: '{claim}'")
    logging.debug(f"Normalized claim: '{normalized_claim}'")
    return normalized_claim


def normalize_w2n(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Converts number words to numbers using the word2number.

    Args:
        items (List[Dict[str, Any]]): List of claim items.

    Returns:
        List[Dict[str, Any]]: Updated list with
        'normalized_claim' added to each item.
    """
    for item in items:
        original_claim = item.get("claim", "")
        normalized_claim = process_claim(original_claim)
        item["normalized_claim"] = normalized_claim
    return items


def is_model_available(model_name: str) -> bool:
    """Indicate whether a spaCy model is available.

    Args:
        model_name (str): Name of the spaCy model.

    Returns:
        bool: True if the model is available, False otherwise.
    """
    try:
        spacy.load(model_name)
        return True
    except Exception:
        return False


def initialize_spacy() -> Optional[Language]:
    """Load the spaCy model and return it.

    Returns:
        Language: The spaCy language model.
    """
    model_name = "en_core_web_trf"
    if not is_model_available(model_name):
        download(model_name)
    try:
        return spacy.load(model_name)
    except OSError as e:
        logging.error(f"Error loading spaCy model: {e}")
        return None


def get_target_entities() -> Set[str]:
    """Return set of entity types we want to classify."""
    return {
        "CARDINAL",
        "MONEY",
        "PERCENT",
        "QUANTITY",
        "TIME",
        "DATE",
        "ORDINAL",
    }


def is_token_mergeable(token, number_words_set: set) -> bool:
    """Determine if token should be part of number merge."""
    return (
        token.like_num
        or token.lower_ in number_words_set
        or token.text in "$£€¥"
    )


def is_token_part_of_merge(token, number_words_set: set) -> bool:
    """Check if next token should be included in merge."""
    return (
        token.like_num
        or token.lower_ in number_words_set
        or token.text.lower() in ["point", ".", ",", "-"]
    )


def find_spans_to_merge(doc, number_words_set: set) -> List[spacy.tokens.Span]:
    """Find all spans of tokens that should be merged."""
    spans_to_merge = []
    i = 0
    while i < len(doc):
        if is_token_mergeable(doc[i], number_words_set):
            start = i
            while (i + 1 < len(doc)) and is_token_part_of_merge(
                doc[i + 1], number_words_set
            ):
                i += 1
            spans_to_merge.append(doc[start : i + 1])
        i += 1
    return spans_to_merge


def extract_entity_info(ent) -> dict:
    """Extract relevant information from an entity."""
    return {
        "text": ent.text,
        "label": ent.label_,
        "start_char": ent.start_char,
        "end_char": ent.end_char,
    }


def process_single_claim(doc: spacy.tokens.Doc, target_entities: set) -> dict:
    """Process a single claim and return token information.

    Args:
        doc (spacy.tokens.Doc): The spaCy processed document.
        target_entities (set): Set of target entity labels.

    Returns:
        dict: Processed information including tokens, ner_tags, and entities.
    """
    # Initialize lists to store tokens and annotations
    tokens = [token.text for token in doc]
    annotations = ["O"] * len(tokens)

    # Process named entities
    for ent in doc.ents:
        if ent.label_ in target_entities:
            # Use ent.start and ent.end to get token indices
            for idx in range(ent.start, ent.end):
                annotations[idx] = ent.label_

    # Prepare the entities list with correct token indices
    entities = []
    for ent in doc.ents:
        if ent.label_ in target_entities:
            entities.append(
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                }
            )

    return {
        "tokens": tokens,
        "ner_tags": annotations,
        "entities": entities,
    }


def classify_entities(
    nlp: Language, items: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Classify entities in claims and add tagging information to each item.

    Args:
        nlp (Language): The loaded spaCy language model.
        items (List[Dict[str, Any]]): List of claim items with
        normalized claims.

    Returns:
        List[Dict[str, Any]]: Updated list with tagging
        information added to each item.
    """
    target_entities = get_target_entities()

    for item in items:
        claim_text = item.get("normalized_claim", item.get("claim", ""))
        doc = nlp(claim_text)
        processed_claim = process_single_claim(doc, target_entities)
        item.update(processed_claim)
    return items


def process_data(
    data: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extract and process statistical claims.

    Extracts statistical claims, normalizes numbers, classifies entities, and
    returns the processed data.

    Args:
        data (List[Dict[str, Any]]): List of claim items.

    Returns:
        tuple: Two lists for false and true claims respectively.
    """
    nlp = initialize_spacy()
    if not nlp:
        raise RuntimeError("Failed to initialize spaCy model.")

    # Filter data into false and true claims with taxonomy_label "statistical"
    false_items = filter_claims(
        data, taxonomy_label="statistical", label="False"
    )
    true_items = filter_claims(data, taxonomy_label="statistical", label="True")

    # Normalize claims
    false_normalized = normalize_w2n(false_items)
    true_normalized = normalize_w2n(true_items)

    # Classify entities and add tagging information
    false_processed = classify_entities(nlp, false_normalized)
    true_processed = classify_entities(nlp, true_normalized)

    return false_processed, true_processed


def main(input_json_path: str, output_json_path1: str, output_json_path2: str):
    """Loads JSON data, processes it, and saves it.

    Args:
        input_json_path (str): Path to the input JSON file.
        output_json_path1 (str): Path to save processed false claims.
        output_json_path2 (str): Path to save processed true claims.
    """
    data = load_json_data(input_json_path)
    false_result, true_result = process_data(data)

    with open(output_json_path1, "w") as outfile1:
        json.dump(false_result, outfile1, ensure_ascii=False, indent=4)

    with open(output_json_path2, "w") as outfile2:
        json.dump(true_result, outfile2, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    input_json_path = ""
    output_json_path1 = ""  # noqa: E501
    output_json_path2 = ""  # noqa: E501
    main(input_json_path, output_json_path1, output_json_path2)
