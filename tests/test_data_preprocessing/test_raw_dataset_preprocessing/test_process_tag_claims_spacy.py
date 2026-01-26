"""Tests for the process_claims module."""

import pytest
import json
import spacy
from unittest.mock import mock_open, patch, MagicMock
from spacy.tokens import Span


from code.data_preprocessing.raw_dataset_preprocessing.process_tag_claims_spacy import (  # noqa: E501
    load_json_data,
    filter_claims,
    get_number_words_set,
    get_multipliers,
    is_number_word,
    word_to_num,
    replace_number_phrases,
    process_claim,
    normalize_w2n,
    is_model_available,
    initialize_spacy,
    get_target_entities,
    is_token_mergeable,
    is_token_part_of_merge,
    find_spans_to_merge,
    extract_entity_info,
    process_single_claim,
    classify_entities,
    process_data,
)

# Test data
MOCK_DATA = [
    {
        "claim": "The GDP increased by five million dollars",
        "taxonomy_label": "statistical",
        "label": "True",
    },
    {
        "claim": "The cost was $10 billion",
        "taxonomy_label": "statistical",
        "label": "False",
    },
    {
        "claim": "This is not statistical",
        "taxonomy_label": "other",
        "label": "True",
    },
]


@pytest.fixture
def mock_spacy_model():
    """Mock spaCy model for testing."""
    model = MagicMock()
    return model


def test_load_json_data():
    """Test loading JSON data from a file."""
    mock_json = json.dumps(MOCK_DATA)
    with patch("builtins.open", mock_open(read_data=mock_json)):
        data = load_json_data("fake_path.json")
        assert len(data) == 3
        assert data[0]["claim"] == "The GDP increased by five million dollars"


def test_filter_claims():
    """Test filtering claims based on taxonomy_label and label."""
    filtered = filter_claims(MOCK_DATA, "statistical", "True")
    assert len(filtered) == 1
    assert filtered[0]["claim"] == "The GDP increased by five million dollars"


def test_get_number_words_set():
    """Test getting the set of number words."""
    number_words = get_number_words_set()
    assert isinstance(number_words, set)
    assert "one" in number_words
    assert "million" in number_words


def test_get_multipliers():
    """Test getting the multipliers dictionary."""
    multipliers = get_multipliers()
    assert multipliers["thousand"] == 1000
    assert multipliers["million"] == 1000000


def test_is_number_word():
    """Test checking if a word is a number word."""
    assert is_number_word("five") is True
    assert is_number_word("123") is True
    assert is_number_word("abc") is False


def test_word_to_num():
    """Test converting a number word to a number."""
    assert word_to_num("five") == 5
    assert word_to_num("123") == 123
    assert word_to_num("abc") is None


def test_replace_number_phrases():
    """Test replacing number phrases with numbers."""
    test_cases = [
        ("five million", "5,000,000"),
        ("$10 billion", "$10,000,000,000"),
        ("no numbers here", "no numbers here"),
    ]
    for input_text, expected in test_cases:
        assert replace_number_phrases(input_text) == expected


def test_process_claim():
    """Test processing a claim to replace number phrases."""
    assert process_claim("five million") == "5,000,000"
    assert process_claim("regular text") == "regular text"


def test_normalize_w2n():
    """Test normalizing claims using word-to-number conversion."""
    test_items = [{"claim": "five million"}]
    result = normalize_w2n(test_items)
    assert result[0]["normalized_claim"] == "5,000,000"


def test_get_target_entities():
    """Test getting the set of target entities."""
    entities = get_target_entities()
    assert "CARDINAL" in entities
    assert "MONEY" in entities


@pytest.mark.parametrize(
    "token_text,expected",
    [("5", True), ("five", True), ("$", True), ("word", False)],
)
def test_is_token_mergeable(mock_spacy_model, token_text, expected):
    """Test checking if a token is mergeable."""
    token = MagicMock()
    token.like_num = token_text.isdigit()
    token.lower_ = token_text.lower()
    token.text = token_text
    number_words = get_number_words_set()
    assert is_token_mergeable(token, number_words) == expected


def test_extract_entity_info():
    """Test extracting entity information."""
    mock_entity = MagicMock()
    mock_entity.text = "5 million"
    mock_entity.label_ = "CARDINAL"
    mock_entity.start_char = 0
    mock_entity.end_char = 8

    result = extract_entity_info(mock_entity)
    assert result["text"] == "5 million"
    assert result["label"] == "CARDINAL"
    assert result["start_char"] == 0
    assert result["end_char"] == 8


def test_is_model_available():
    """Test checking if a spaCy model is available."""
    # Test with a valid model
    with patch("spacy.load") as mock_load:
        mock_load.return_value = True
        assert is_model_available("en_core_web_trf") is True
        mock_load.assert_called_once_with("en_core_web_trf")

    # Test with an invalid model
    with patch("spacy.load") as mock_load:
        mock_load.side_effect = Exception("Model not found")
        assert is_model_available("invalid_model") is False
        mock_load.assert_called_once_with("invalid_model")


def test_initialize_spacy():
    """Test initializing the spaCy model."""
    # Test successful initialization
    with (
        patch(
            "code.data_preprocessing.raw_dataset_preprocessing.process_tag_claims_spacy.is_model_available"  # noqa: E501
        ) as mock_available,
        patch("spacy.load") as mock_load,
    ):
        mock_available.return_value = True
        mock_load.return_value = "mock_model"

        result = initialize_spacy()
        assert result == "mock_model"
        mock_load.assert_called_once_with("en_core_web_trf")
        mock_available.assert_called_once()

    # Test when model needs to be downloaded
    with (
        patch(
            "code.data_preprocessing.raw_dataset_preprocessing.process_tag_claims_spacy.is_model_available"  # noqa: E501
        ) as mock_available,
        patch(
            "code.data_preprocessing.raw_dataset_preprocessing.process_tag_claims_spacy.download"  # noqa: E501
        ) as mock_download,
        patch("spacy.load") as mock_load,
    ):
        mock_available.return_value = False
        mock_load.return_value = "mock_model"

        result = initialize_spacy()
        assert result == "mock_model"
        mock_download.assert_called_once_with("en_core_web_trf")
        mock_load.assert_called_once_with("en_core_web_trf")
        mock_available.assert_called_once()

    # Test when loading fails
    with (
        patch(
            "code.data_preprocessing.raw_dataset_preprocessing.process_tag_claims_spacy.is_model_available"  # noqa: E501
        ) as mock_available,
        patch("spacy.load") as mock_load,
    ):
        mock_available.return_value = True
        mock_load.side_effect = OSError("Model loading failed")

        result = initialize_spacy()
        assert result is None
        mock_load.assert_called_once_with("en_core_web_trf")


@pytest.mark.parametrize(
    "token_text,token_like_num,expected",
    [
        ("5", True, True),
        ("five", False, True),
        ("point", False, True),
        (".", False, True),
        (",", False, True),
        ("-", False, True),
        ("word", False, False),
    ],
)
def test_is_token_part_of_merge(
    mock_spacy_model, token_text, token_like_num, expected
):
    """Test checking if a token is part of a merge."""
    token = MagicMock()
    token.like_num = token_like_num
    token.lower_ = token_text.lower()
    token.text = token_text
    number_words = get_number_words_set()
    assert is_token_part_of_merge(token, number_words) == expected


def test_process_data(mock_spacy_model):
    """Test processing data to classify entities."""
    test_data = [
        {
            "claim": "The GDP is 5 million",
            "taxonomy_label": "statistical",
            "label": "True",
        },
        {
            "claim": "Cost was $10 billion",
            "taxonomy_label": "statistical",
            "label": "False",
        },
        {
            "claim": "Not statistical",
            "taxonomy_label": "other",
            "label": "True",
        },
    ]

    # Mock initialize_spacy
    with patch(
        "code.data_preprocessing.raw_dataset_preprocessing.process_tag_claims_spacy.initialize_spacy"  # noqa: E501
    ) as mock_init:
        mock_init.return_value = mock_spacy_model

        # Process data
        false_items, true_items = process_data(test_data)

        # Verify results
        assert len(false_items) == 1
        assert len(true_items) == 1
        assert false_items[0]["claim"] == "Cost was $10 billion"
        assert true_items[0]["claim"] == "The GDP is 5 million"

        # Verify normalized claims were added
        assert "normalized_claim" in false_items[0]
        assert "normalized_claim" in true_items[0]

        # Verify entities were classified
        assert "tokens" in false_items[0]
        assert "ner_tags" in false_items[0]
        assert "entities" in false_items[0]
        assert "tokens" in true_items[0]
        assert "ner_tags" in true_items[0]
        assert "entities" in true_items[0]


def test_process_data_failed_initialization():
    """Test handling of failed spaCy initialization."""
    # Test handling of failed spaCy initialization
    with patch(
        "code.data_preprocessing.raw_dataset_preprocessing.process_tag_claims_spacy.initialize_spacy"  # noqa: E501
    ) as mock_init:
        mock_init.return_value = None

        with pytest.raises(RuntimeError) as exc_info:
            process_data([])

        assert str(exc_info.value) == "Failed to initialize spaCy model."


def test_classify_entities():
    """Test classifying entities in a document."""
    # Mock spaCy model and document
    nlp = MagicMock()
    doc = MagicMock()

    # Mock entity
    entity = MagicMock()
    entity.text = "5 million"
    entity.label_ = "CARDINAL"
    entity.start = 0
    entity.end = 2
    entity.start_char = 0
    entity.end_char = 8

    doc.ents = [entity]
    doc.__iter__ = lambda self: iter(
        [MagicMock(text="5"), MagicMock(text="million")]
    )
    nlp.return_value = doc

    # Test data
    items = [
        {
            "claim": "The number is 5 million",
            "normalized_claim": "The number is 5,000,000",
        }
    ]

    # Call function
    result = classify_entities(nlp, items)

    # Verify results
    assert len(result) == 1
    assert "tokens" in result[0]
    assert "ner_tags" in result[0]
    assert "entities" in result[0]

    # Check entity info
    entity_info = result[0]["entities"][0]
    assert entity_info["text"] == "5 million"
    assert entity_info["label"] == "CARDINAL"
    assert entity_info["start_char"] == 0
    assert entity_info["end_char"] == 8


def test_find_spans_to_merge():
    """Test finding spans to merge in a document."""
    nlp = spacy.blank("en")
    doc = nlp("5 million and 10 billion")

    number_words = get_number_words_set()
    spans = find_spans_to_merge(doc, number_words)

    assert len(spans) == 2
    assert spans[0].text == "5 million"
    assert spans[1].text == "10 billion"


def test_process_single_claim():
    """Test processing a single claim with named entities."""
    nlp = spacy.blank("en")
    doc = nlp("The GDP is 5 million")

    # Token indices for the entity "5 million"
    start = 3  # Index of token "5"
    end = 5  # Index after "million" (end index is exclusive)

    # Create a Span for the entity and assign a label
    span = Span(doc, start, end, label="CARDINAL")

    # Assign the span to doc.ents
    doc.ents = [span]

    result = process_single_claim(doc, {"CARDINAL"})

    # Check basic output structure
    assert "tokens" in result
    assert "ner_tags" in result
    assert "entities" in result
    assert len(result["entities"]) == 1
    assert result["entities"][0]["text"] == "5 million"
