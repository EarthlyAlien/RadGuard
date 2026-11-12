"""Shared pytest fixtures."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

CASES_PATH = Path(__file__).resolve().parents[1] / "data" / "synthetic_cases.json"


@pytest.fixture(scope="session")
def synthetic_cases() -> list[dict[str, object]]:
    """Load the synthetic findings/impression pairs."""
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def client() -> Iterator[TestClient]:
    """HTTP client bound to the RadGuard FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client
