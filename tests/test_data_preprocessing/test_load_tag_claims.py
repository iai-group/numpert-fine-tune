"""Tests for the load_tag_claims module."""
import json
from unittest.mock import MagicMock, mock_open, patch
from numpert.data_preprocessing.load_tag_claims import (
    load_json_data,
    url_true_claim_statistical,
    initialize_spacy,
    classify_entities,
)


def test_load_json_data():
    """Test that the function loads the json data correctly."""
    mock_path = "mock_path.json"
    mock_data = {"key": "value"}

    with patch("builtins.open", mock_open(read_data=json.dumps(mock_data))):
        results = load_json_data(mock_path)

    assert results == mock_data


def test_url_true_statistical():
    """Test that the function returns the correct data."""
    mock_data = [
        {
            "taxonomy_label": "statistical",
            "label": "True",
            "url": "http://example.com/1",
            "claim": "Claim 1",
        },
        {
            "taxonomy_label": "non-statistical",
            "label": "True",
            "url": "http://example.com/2",
            "claim": "Claim 2",
        },
        {
            "taxonomy_label": "statistical",
            "label": "False",
            "url": "http://example.com/3",
            "claim": "Claim 3",
        },
        {
            "taxonomy_label": "statistical",
            "label": "True",
            "url": "http://example.com/4",
            "claim": "Claim 4",
        },
    ]

    expected = {
        "http://example.com/1": "Claim 1",
        "http://example.com/4": "Claim 4",
    }

    result = url_true_claim_statistical(mock_data)

    assert result == expected


def test_initialize_spacy():
    """Test that the function initializes SpaCy correctly."""
    with patch("spacy.load") as mock_load:
        # Mock the return value of spacy.load
        mock_nlp = object()
        mock_load.return_value = mock_nlp

        # Call the function to test
        nlp = initialize_spacy()

        # Assert spacy.load was called with the correct model name
        mock_load.assert_called_with("en_core_web_trf")

        # Assert the returned nlp object is as expected
        assert nlp == mock_nlp


def test_classify_entities():
    """Tests classify_entities processes text, returns expected annotations."""
    # Mock the SpaCy NLP pipeline
    with patch(
        "numpert.data_preprocessing.load_tag_claims.initialize_spacy"
    ) as mock_initialize_spacy:
        mock_nlp = MagicMock()
        mock_doc = MagicMock()

        # Define mock tokens with entity annotations
        tokens = []
        token_texts = ["The", "are", "7", "horses", "in", "the", "field"]
        token_ent_types = ["O", "O", "CARDINAL", "O", "O", "O", "O"]

        for text, ent_type in zip(token_texts, token_ent_types):
            token = MagicMock()
            token.text = text
            token.ent_type_ = ent_type
            tokens.append(token)

        # Simulate the document object returned by SpaCy
        mock_doc.__iter__.return_value = tokens
        mock_doc.text = "The are 7 horses in the field"

        # Set up the mock NLP pipeline to return the mock document
        mock_nlp.return_value = mock_doc
        mock_initialize_spacy.return_value = mock_nlp

        # Input data and expected output
        input_data = {"someurl": "The are 7 horses in the field"}
        expected_output = {
            "someurl": (
                "The are 7 horses in the field",
                ["0", "0", "CARDINAL", "0", "0", "0", "0"],
            )
        }

        # Call the function under test
        nlp = initialize_spacy()
        results = classify_entities(nlp, input_data)

        # Assert the results match the expected output
        assert results == expected_output
