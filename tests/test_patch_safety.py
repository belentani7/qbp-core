"""Prevent duplicate repairs and invalid manifest mutations."""

import pytest

from qbp.errors import PatchApplicationError
from qbp.patch import CorrectionPatch, auto_correct
from qbp.qa import QAReport


def test_stale_report_applies_only_once(sample_qa_report_failing, sample_manifest):
    report = QAReport.parse(sample_qa_report_failing)
    fixed, patches = auto_correct(report, sample_manifest, max_iterations=20)
    assert len(patches) == 1
    assert len(fixed.get("_patch_history", [])) == 1
    assert len([c for c in fixed["controls"] if c["type"] == "reference"]) == 1


def test_invalid_candidate_is_discarded(sample_qa_report_failing, sample_manifest):
    report = QAReport.parse(sample_qa_report_failing)
    result, patches = auto_correct(
        report, sample_manifest, validator=lambda candidate: ["invalid format"]
    )
    assert patches == []
    assert result == sample_manifest


def test_reassessment_allows_actual_second_pass(sample_qa_report_failing,
                                                 sample_qa_report_passing,
                                                 sample_manifest):
    result, patches = auto_correct(
        QAReport.parse(sample_qa_report_failing),
        sample_manifest,
        max_iterations=5,
        reassess=lambda manifest: QAReport.parse(sample_qa_report_passing),
    )
    assert len(patches) == 1
    assert result["sections"]["atmosphere"]["mood"] == "dramatic"


def test_invalid_path_is_rejected(sample_manifest):
    patch = CorrectionPatch(operations=[
        {"op": "modify", "path": "", "value": 12}
    ])
    with pytest.raises(PatchApplicationError, match="Invalid manifest path"):
        patch.apply(sample_manifest)


def test_invalid_path_does_not_mutate_original(sample_manifest):
    patch = CorrectionPatch(operations=[
        {"op": "modify", "path": "sections.identity.age.child", "value": 12}
    ])
    with pytest.raises(PatchApplicationError):
        patch.apply(sample_manifest)
    assert sample_manifest["sections"]["identity"]["age"] == 30


def test_negative_iterations_rejected(sample_qa_report_failing, sample_manifest):
    with pytest.raises(ValueError):
        auto_correct(QAReport.parse(sample_qa_report_failing),
                     sample_manifest, max_iterations=-1)
