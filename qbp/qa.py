"""QA Report parsing, scoring, and analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import QAReportParseError

# ── QA dimension weights ──────────────────────────────────────────────────────

DEFAULT_DIMENSION_WEIGHTS: dict[str, float] = {
    "identity_fidelity": 0.30,
    "temporal_consistency": 0.25,
    "pbr_validity": 0.25,
    "aesthetic_coherence": 0.20,
}


class QAReport:
    """Parsed QA report with scoring helpers.

    Example
    -------
    >>> report = QAReport.parse(report_dict)
    >>> if report.needs_regeneration():
    ...     print(report.summary())
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data
        self._checks: list[dict[str, Any]] = data.get("checks", [])
        self._dimensions: dict[str, Any] = data.get("dimensions", {})
        self._meta: dict[str, Any] = data.get("meta", {})
        self._dimension_weights = data.get(
            "dimension_weights", DEFAULT_DIMENSION_WEIGHTS
        )

    # ── parsing ────────────────────────────────────────────────────────────

    @classmethod
    def parse(cls, report_json: dict[str, Any] | str | Path) -> QAReport:
        """Parse a QA report from a dict, JSON string, or file path.

        Parameters
        ----------
        report_json:
            A dict, a raw JSON string, or a ``Path`` to a ``.json`` file.

        Returns
        -------
        QAReport
        """
        if isinstance(report_json, dict):
            data = report_json
        elif isinstance(report_json, str):
            # Try as JSON string first (starts with '{'), then as file path
            stripped = report_json.strip()
            if stripped.startswith("{"):
                try:
                    data = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise QAReportParseError(f"Invalid JSON string: {exc}") from exc
            else:
                p = Path(report_json)
                if not p.exists():
                    raise QAReportParseError(f"Report file not found: {p}")
                with p.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
        elif isinstance(report_json, Path):
            if not report_json.exists():
                raise QAReportParseError(f"Report file not found: {report_json}")
            with report_json.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        else:
            raise QAReportParseError(
                f"Expected dict, str, or Path, got {type(report_json).__name__}"
            )

        # Basic structural validation
        if not isinstance(data, dict):
            raise QAReportParseError("Report must be a JSON object (dict)")
        if "checks" not in data:
            raise QAReportParseError("Report is missing required 'checks' key")

        return cls(data)

    # ── scoring ────────────────────────────────────────────────────────────

    def overall_score(self) -> float:
        """Compute the weighted overall score in ``[0, 1]``.

        Score is the weighted sum of dimension scores (each ``[0, 1]``).
        Falls back to check-based scoring when dimension scores are absent.
        """
        # Prefer dimension-level scores if present
        if self._dimensions:
            total_weight = 0.0
            weighted_sum = 0.0
            for dim, weight in self._dimension_weights.items():
                dim_data = self._dimensions.get(dim)
                if dim_data is None:
                    continue
                score = dim_data.get("score") if isinstance(dim_data, dict) else dim_data
                if isinstance(score, (int, float)):
                    weighted_sum += float(score) * weight
                    total_weight += weight
            if total_weight > 0:
                return round(weighted_sum / total_weight, 4)

        # Fallback: derive score from individual checks
        return self._check_based_score()

    def _check_based_score(self) -> float:
        """Derive a score from individual check results."""
        if not self._checks:
            return 0.0
        passed = sum(1 for c in self._checks if c.get("passed", False))
        return round(passed / len(self._checks), 4)

    def needs_regeneration(self, threshold: float = 0.7) -> bool:
        """Return ``True`` if the overall score falls below *threshold*."""
        return self.overall_score() < threshold

    def get_failures(self) -> list[dict[str, Any]]:
        """Return all checks where ``passed`` is ``False``."""
        return [c for c in self._checks if not c.get("passed", False)]

    def summary(self) -> str:
        """Human-readable one-line summary of the report."""
        score = self.overall_score()
        n_total = len(self._checks)
        n_failed = len(self.get_failures())
        asset_id = self._meta.get("asset_id", "unknown")
        return (
            f"QA Report [{asset_id}]: "
            f"score={score:.2f}, "
            f"checks={n_total - n_failed}/{n_total} passed, "
            f"regen={'YES' if self.needs_regeneration() else 'no'}"
        )

    @property
    def asset_id(self) -> str | None:
        """Asset identifier from report metadata."""
        return self._meta.get("asset_id")

    @property
    def raw(self) -> dict[str, Any]:
        """Access the underlying report dict."""
        return self._data

    def __repr__(self) -> str:
        return f"QAReport(score={self.overall_score():.2f}, checks={len(self._checks)})"
