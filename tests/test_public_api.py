"""Regression tests for the package-root public API."""

from pathlib import Path

import pytest

def test_readme_public_api_imports() -> None:
    from qbp import BindingManifest, ControlSignal, CorrectionPatch, QAReport, auto_correct

    assert BindingManifest.__name__ == "BindingManifest"
    assert ControlSignal.__name__ == "ControlSignal"
    assert CorrectionPatch.__name__ == "CorrectionPatch"
    assert QAReport.__name__ == "QAReport"
    assert callable(auto_correct)


def test_public_api_declares_core_symbols() -> None:
    import qbp

    for name in (
        "BindingManifest",
        "ControlSignal",
        "CorrectionPatch",
        "QAReport",
        "auto_correct",
    ):
        assert name in qbp.__all__


def test_readme_manifest_round_trip(tmp_path: Path) -> None:
    from qbp import BindingManifest, QAReport, auto_correct

    manifest = BindingManifest.create(
        asset_type="character",
        brief="A reference character",
        identity={"name": "Original"},
        geometry={"height": 180},
        motion={"idle": "standing"},
        atmosphere={"mood": "neutral"},
    )
    manifest["controls"] = BindingManifest.generate_controls(["depth"], [0.8])
    obj = BindingManifest.from_dict(manifest)
    manifest["sections"]["identity"]["name"] = "Changed"
    assert obj.data["sections"]["identity"]["name"] == "Original"

    path = tmp_path / "manifest.json"
    obj.to_json(path)
    loaded = BindingManifest.from_json(path)
    assert loaded.data == obj.data
    assert BindingManifest.validate(loaded.data) == []

    report = QAReport.parse({"checks": [{"dimension": "identity_fidelity", "passed": True}]})
    corrected, patches = auto_correct(report, loaded.data)
    assert patches == []
    assert corrected == loaded.data


@pytest.mark.parametrize("invalid", [None, [], "not a manifest", 42])
def test_public_validator_returns_errors_for_non_objects(invalid: object) -> None:
    from qbp import BindingManifest, ManifestValidationError

    errors = BindingManifest.validate(invalid)
    assert errors and "dict" in errors[0]
    with pytest.raises(ManifestValidationError):
        BindingManifest.validate_strict(invalid)


def test_validator_rejects_unhashable_asset_type() -> None:
    from qbp import BindingManifest

    manifest = BindingManifest.create(
        asset_type="character",
        brief="A reference character",
        identity={},
        geometry={},
        motion={},
        atmosphere={},
    )
    manifest["asset_type"] = ["character"]
    assert any("asset_type" in error for error in BindingManifest.validate(manifest))
