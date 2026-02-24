"""Pytest fixtures for raw dataset preprocessing tests."""

import pytest
from itertools import count
from src.data_preprocessing.raw_dataset_preprocessing import (
    create_binary_dataset,
)


@pytest.fixture(autouse=True)
def reset_claim_counter():
    """Reset the global claim counter before each test."""
    create_binary_dataset._claim_counter = count(start=1)
    yield
