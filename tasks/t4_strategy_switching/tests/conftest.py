"""Pytest fixtures for T4 strategy switching tests."""

import sys
from pathlib import Path

# Add tests dir to path so helpers.py is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest
from helpers import load_all_problems


@pytest.fixture
def all_problems():
    return load_all_problems()
