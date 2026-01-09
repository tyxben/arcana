"""Pytest configuration and fixtures for Arcana tests."""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_trace_dir() -> Path:
    """Create a temporary directory for trace files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_run_id() -> str:
    """Provide a sample run ID for testing."""
    return "test-run-001"
