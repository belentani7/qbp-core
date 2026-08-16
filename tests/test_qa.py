"""Unit tests for qbp.qa."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from qbp.qa import QAReport, DEFAULT_DIMENSION_WEIGHTS
from qbp.errors import QAReportParseError


class TestParse:
    def test_parse_dict(self, sample_qa_report_passing: dict[str, Any]) -> None:
        report = QAReport.parse(sample_qa_report_passing)
        assert len(report._checks) == 4

    def test_parse_json_string(self, sample_qa_report_passing: dict[str, Any]) -> None:
        import json
        report = QAReport.parse(json.dumps(sample_qa_report_passing))
        assert report.overall_score() > 0

    def test_parse_file(self, sample_qa_report_passing: dict[str, Any], tmp_path: Path) -> None:
        import json
        p = tmp_path / "report.json"
        p.write_text(json.dumps(sample_qa_report_passing))
        report = QAReport.parse(p)
        assert report.asset_id == "test-asset-001"

    def test_parse_file_not_found(self) -> None:
        with pytest.raises(QAReportParseError, match="not found"):
            QAReport.parse("/nonexistent/report.json")

    def test_parse_invalid_type(self) -> None:
        with pytest.raises(QAReportParseError, match="Expected dict"):
            QAReport.parse(12345)  # type: ignore[arg-type]

    def test_parse_missing_checks(self) -> None:
        with pytest.raises(QAReportParseError, match="missing required 'checks'"):
            QAReport.parse({"meta": {}})

    def test_parse_not_dict(self) -> None:
        with pytest.raises(QAReportParseError, match="Expected dict"):
            QAReport.parse([1, 2, 3])  # type: ignore[arg-type]


class TestOverallScore:
    def test_passing_report(self, sample_qa_report_passing: dict[str, Any]) -> None:
        report = QAReport.parse(sample_qa_report_passing)
        score = report.overall_score()
        assert 0.8 < score <= 1.0

    def test_failing_report(self, sample_qa_report_failing: dict[str, Any]) -> None:
        report = QAReport.parse(sample_qa_report_failing)
        score = report.overall_score()
        assert score < 0.7

    def test_fallback_check_based(self) -> None:
        """When no dimension scores, falls back to check pass rate."""
        report = QAReport.parse({
            "checks": [
                {"passed": True},
                {"passed": True},
                {"passed": False},
            ],
        })
        assert abs(report.overall_score() - 2 / 3) < 0.01

    def test_empty_checks(self) -> None:
        report = QAReport.parse({"checks": []})
        assert report.overall_score() == 0.0

    def test_empty_report_no_dimensions(self) -> None:
        report = QAReport.parse({
            "checks": [{"passed": True}, {"passed": False}],
            "dimensions": {},
        })
        assert report.overall_score() == 0.5


class TestNeedsRegeneration:
    def test_below_threshold(self, sample_qa_report_failing: dict[str, Any]) -> None:
        report = QAReport.parse(sample_qa_report_failing)
        assert report.needs_regeneration(threshold=0.7) is True

    def test_above_threshold(self, sample_qa_report_passing: dict[str, Any]) -> None:
        report = QAReport.parse(sample_qa_report_passing)
        assert report.needs_regeneration(threshold=0.7) is False

    def test_custom_threshold(self, sample_qa_report_passing: dict[str, Any]) -> None:
        report = QAReport.parse(sample_qa_report_passing)
        # Even passing report may be below 0.95
        result = report.needs_regeneration(threshold=0.99)
        assert isinstance(result, bool)


class TestGetFailures:
    def test_all_pass(self, sample_qa_report_passing: dict[str, Any]) -> None:
        report = QAReport.parse(sample_qa_report_passing)
        assert report.get_failures() == []

    def test_some_fail(self, sample_qa_report_failing: dict[str, Any]) -> None:
        report = QAReport.parse(sample_qa_report_failing)
        failures = report.get_failures()
        assert len(failures) == 2
        assert all(not f["passed"] for f in failures)


class TestSummary:
    def test_summary_format(self, sample_qa_report_passing: dict[str, Any]) -> None:
        report = QAReport.parse(sample_qa_report_passing)
        s = report.summary()
        assert "QA Report" in s
        assert "score=" in s
        assert "checks=" in s

    def test_summary_with_asset_id(self, sample_qa_report_passing: dict[str, Any]) -> None:
        report = QAReport.parse(sample_qa_report_passing)
        s = report.summary()
        assert "test-asset-001" in s

    def test_summary_regen_yes(self, sample_qa_report_failing: dict[str, Any]) -> None:
        report = QAReport.parse(sample_qa_report_failing)
        s = report.summary()
        assert "regen=YES" in s


class TestProperties:
    def test_asset_id(self, sample_qa_report_passing: dict[str, Any]) -> None:
        report = QAReport.parse(sample_qa_report_passing)
        assert report.asset_id == "test-asset-001"

    def test_asset_id_missing(self) -> None:
        report = QAReport.parse({"checks": [], "meta": {}})
        assert report.asset_id is None

    def test_raw(self, sample_qa_report_passing: dict[str, Any]) -> None:
        report = QAReport.parse(sample_qa_report_passing)
        assert report.raw is sample_qa_report_passing

    def test_repr(self, sample_qa_report_passing: dict[str, Any]) -> None:
        report = QAReport.parse(sample_qa_report_passing)
        assert "QAReport" in repr(report)

    def test_dimension_weights_default(self) -> None:
        assert "identity_fidelity" in DEFAULT_DIMENSION_WEIGHTS
        total = sum(DEFAULT_DIMENSION_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01
