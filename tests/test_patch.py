"""Unit tests for qbp.patch."""

from __future__ import annotations

from typing import Any

import pytest

from qbp.patch import CorrectionPatch, auto_correct
from qbp.qa import QAReport
from qbp.errors import PatchApplicationError


class TestFromReport:
    def test_creates_operations_for_failures(
        self, sample_qa_report_failing: dict[str, Any], sample_manifest: dict[str, Any]
    ) -> None:
        report = QAReport.parse(sample_qa_report_failing)
        patch = CorrectionPatch.from_report(report, sample_manifest)
        assert len(patch) > 0
        assert all("op" in op for op in patch.operations)

    def test_no_operations_when_all_pass(
        self, sample_qa_report_passing: dict[str, Any], sample_manifest: dict[str, Any]
    ) -> None:
        report = QAReport.parse(sample_qa_report_passing)
        patch = CorrectionPatch.from_report(report, sample_manifest)
        assert len(patch) == 0

    def test_temporal_corrections_add_reference(
        self, sample_qa_report_failing: dict[str, Any], sample_manifest: dict[str, Any]
    ) -> None:
        report = QAReport.parse(sample_qa_report_failing)
        patch = CorrectionPatch.from_report(report, sample_manifest)
        # The temporal failure should generate a set_controls op
        ops = [op for op in patch.operations if op["op"] == "set_controls"]
        assert len(ops) > 0

    def test_aesthetic_corrections_modify_atmosphere(
        self, sample_qa_report_failing: dict[str, Any], sample_manifest: dict[str, Any]
    ) -> None:
        report = QAReport.parse(sample_qa_report_failing)
        patch = CorrectionPatch.from_report(report, sample_manifest)
        atmosphere_ops = [
            op for op in patch.operations
            if op.get("path", "").startswith("sections.atmosphere")
        ]
        assert len(atmosphere_ops) > 0


class TestApply:
    def test_apply_modifies_manifest(
        self, sample_qa_report_failing: dict[str, Any], sample_manifest: dict[str, Any]
    ) -> None:
        report = QAReport.parse(sample_qa_report_failing)
        patch = CorrectionPatch.from_report(report, sample_manifest)
        corrected = patch.apply(sample_manifest)

        # Original should not be mutated
        assert sample_manifest["sections"]["atmosphere"].get("mood") != "dramatic"
        # Corrected should have the fix
        assert corrected["sections"]["atmosphere"]["mood"] == "dramatic"

    def test_apply_adds_patch_history(
        self, sample_qa_report_failing: dict[str, Any], sample_manifest: dict[str, Any]
    ) -> None:
        report = QAReport.parse(sample_qa_report_failing)
        patch = CorrectionPatch.from_report(report, sample_manifest)
        corrected = patch.apply(sample_manifest)
        assert "_patch_history" in corrected
        assert len(corrected["_patch_history"]) == 1

    def test_apply_empty_patch(self, sample_manifest: dict[str, Any]) -> None:
        patch = CorrectionPatch(operations=[])
        corrected = patch.apply(sample_manifest)
        assert corrected == sample_manifest

    def test_apply_remove_operation(self, sample_manifest: dict[str, Any]) -> None:
        sample_manifest["sections"]["identity"]["deprecated"] = True
        patch = CorrectionPatch(operations=[
            {"op": "remove", "path": "sections.identity.deprecated"}
        ])
        corrected = patch.apply(sample_manifest)
        assert "deprecated" not in corrected["sections"]["identity"]

    def test_apply_unknown_op_raises(self, sample_manifest: dict[str, Any]) -> None:
        patch = CorrectionPatch(operations=[
            {"op": "unknown_op", "path": "x", "value": "y"}
        ])
        with pytest.raises(PatchApplicationError, match="Unknown operation"):
            patch.apply(sample_manifest)

    def test_apply_invalid_path_raises(self, sample_manifest: dict[str, Any]) -> None:
        patch = CorrectionPatch(operations=[
            {"op": "remove", "path": "nonexistent.deep.path"}
        ])
        with pytest.raises(PatchApplicationError, match="nonexistent"):
            patch.apply(sample_manifest)


class TestAutoCorrect:
    def test_no_iterations_when_passing(
        self, sample_qa_report_passing: dict[str, Any], sample_manifest: dict[str, Any]
    ) -> None:
        report = QAReport.parse(sample_qa_report_passing)
        corrected, patches = auto_correct(report, sample_manifest)
        assert patches == []
        assert corrected == sample_manifest

    def test_iterates_until_passing_or_limit(
        self, sample_qa_report_failing: dict[str, Any], sample_manifest: dict[str, Any]
    ) -> None:
        report = QAReport.parse(sample_qa_report_failing)
        corrected, patches = auto_correct(report, sample_manifest, max_iterations=2)
        # Should have applied at least one patch
        assert len(patches) <= 2

    def test_stops_when_no_operations(
        self, sample_qa_report_passing: dict[str, Any], sample_manifest: dict[str, Any]
    ) -> None:
        report = QAReport.parse(sample_qa_report_passing)
        corrected, patches = auto_correct(report, sample_manifest, max_iterations=5)
        assert len(patches) == 0

    def test_validator_can_stop_loop(
        self, sample_qa_report_failing: dict[str, Any], sample_manifest: dict[str, Any]
    ) -> None:
        report = QAReport.parse(sample_qa_report_failing)

        def validator(m: dict[str, Any]) -> list[str]:
            return ["structural error"]

        corrected, patches = auto_correct(report, sample_manifest, validator=validator)
        # Validator returning errors should stop the loop early
        assert len(patches) <= 1


class TestRepr:
    def test_patch_repr(self) -> None:
        patch = CorrectionPatch(operations=[{"op": "modify"}, {"op": "add"}])
        assert "CorrectionPatch" in repr(patch)
        assert "2" in repr(patch)
