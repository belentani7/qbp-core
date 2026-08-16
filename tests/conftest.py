"""Shared pytest fixtures for the QBP test suite."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture()
def sample_manifest() -> dict[str, Any]:
    """A minimal valid binding manifest."""
    return {
        "version": "1.0",
        "id": "00000000-0000-0000-0000-000000000001",
        "created_at": "2026-01-15T12:00:00+00:00",
        "asset_type": "character",
        "brief": "A test character for unit testing",
        "tags": ["test", "unit"],
        "sections": {
            "identity": {"name": "TestSubject", "age": 30},
            "geometry": {"height": 170, "build": "average"},
            "motion": {"idle": "standing", "walk": "normal"},
            "atmosphere": {"mood": "neutral", "lighting": "studio"},
        },
        "controls": [],
    }


@pytest.fixture()
def sample_manifest_with_controls(sample_manifest: dict[str, Any]) -> dict[str, Any]:
    """Manifest pre-loaded with control signals."""
    m = dict(sample_manifest)
    m["controls"] = [
        {"type": "depth", "weight": 0.8},
        {"type": "canny", "weight": 0.6},
    ]
    return m


@pytest.fixture()
def sample_qa_report_passing() -> dict[str, Any]:
    """A QA report where all checks pass."""
    return {
        "checks": [
            {"dimension": "identity_fidelity", "passed": True, "score": 0.95},
            {"dimension": "temporal_consistency", "passed": True, "score": 0.88},
            {"dimension": "pbr_validity", "passed": True, "score": 0.92},
            {"dimension": "aesthetic_coherence", "passed": True, "score": 0.85},
        ],
        "dimensions": {
            "identity_fidelity": {"score": 0.95},
            "temporal_consistency": {"score": 0.88},
            "pbr_validity": {"score": 0.92},
            "aesthetic_coherence": {"score": 0.85},
        },
        "meta": {"asset_id": "test-asset-001"},
    }


@pytest.fixture()
def sample_qa_report_failing() -> dict[str, Any]:
    """A QA report with failures requiring regeneration."""
    return {
        "checks": [
            {"dimension": "identity_fidelity", "passed": True, "score": 0.91},
            {
                "dimension": "temporal_consistency",
                "passed": False,
                "score": 0.35,
                "suggestion": "Frames exhibit significant flicker",
                "path": "sections.motion",
                "details": {
                    "motion_params": {"smoothing": 0.8},
                    "add_reference_control": True,
                },
            },
            {"dimension": "pbr_validity", "passed": True, "score": 0.80},
            {
                "dimension": "aesthetic_coherence",
                "passed": False,
                "score": 0.50,
                "suggestion": "Mood mismatch with lighting",
                "path": "sections.atmosphere",
                "details": {
                    "atmosphere_fixes": {"mood": "dramatic", "lighting": "cinematic"},
                    "score": 0.50,
                },
            },
        ],
        "dimensions": {
            "identity_fidelity": {"score": 0.91},
            "temporal_consistency": {"score": 0.35},
            "pbr_validity": {"score": 0.80},
            "aesthetic_coherence": {"score": 0.50},
        },
        "meta": {"asset_id": "test-asset-002"},
    }


@pytest.fixture()
def tmp_json(tmp_path: Path):
    """Factory fixture: write a dict to a temp JSON file, return its Path."""

    def _write(data: dict[str, Any], name: str = "test.json") -> Path:
        p = tmp_path / name
        with p.open("w", encoding="utf-8") as fh:
            json.dump(data, fh)
        return p

    return _write
