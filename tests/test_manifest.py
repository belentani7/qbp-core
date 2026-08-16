"""Unit tests for qbp.manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from qbp.manifest import BindingManifest, MANIFEST_VERSION, REQUIRED_ASSET_TYPES
from qbp.controls import ALL_CONTROL_TYPES
from qbp.errors import ManifestValidationError, ManifestNotFoundError


class TestCreate:
    def test_basic_creation(self, sample_manifest: dict[str, Any]) -> None:
        m = BindingManifest.create(
            asset_type="character",
            brief="A test character for unit testing",
            identity={"name": "TestSubject", "age": 30},
            geometry={"height": 170, "build": "average"},
            motion={"idle": "standing", "walk": "normal"},
            atmosphere={"mood": "neutral", "lighting": "studio"},
        )
        assert m["version"] == MANIFEST_VERSION
        assert m["asset_type"] == "character"
        assert "id" in m
        assert len(m["id"]) == 36  # UUID
        assert m["tags"] == []
        assert m["controls"] == []

    def test_with_tags_and_controls(self) -> None:
        m = BindingManifest.create(
            asset_type="prop",
            brief="A magical sword",
            identity={"name": "Excalibur"},
            geometry={"length": 100},
            motion={},
            atmosphere={},
            tags=["weapon", "legendary"],
            controls=[{"type": "depth", "weight": 0.7}],
        )
        assert m["tags"] == ["weapon", "legendary"]
        assert len(m["controls"]) == 1

    def test_all_asset_types(self) -> None:
        for at in REQUIRED_ASSET_TYPES:
            m = BindingManifest.create(
                asset_type=at,
                brief=f"Test {at}",
                identity={},
                geometry={},
                motion={},
                atmosphere={},
            )
            assert m["asset_type"] == at


class TestValidate:
    def test_valid_manifest(self, sample_manifest: dict[str, Any]) -> None:
        errors = BindingManifest.validate(sample_manifest)
        assert errors == []

    def test_invalid_version(self, sample_manifest: dict[str, Any]) -> None:
        sample_manifest["version"] = "0.0"
        errors = BindingManifest.validate(sample_manifest)
        assert any("version" in e for e in errors)

    def test_invalid_asset_type(self, sample_manifest: dict[str, Any]) -> None:
        sample_manifest["asset_type"] = "invalid_type"
        errors = BindingManifest.validate(sample_manifest)
        assert any("asset_type" in e for e in errors)

    def test_empty_brief(self, sample_manifest: dict[str, Any]) -> None:
        sample_manifest["brief"] = ""
        errors = BindingManifest.validate(sample_manifest)
        assert any("Brief" in e for e in errors)

    def test_long_brief(self, sample_manifest: dict[str, Any]) -> None:
        sample_manifest["brief"] = "x" * 2001
        errors = BindingManifest.validate(sample_manifest)
        assert any("2000" in e for e in errors)

    def test_missing_section(self, sample_manifest: dict[str, Any]) -> None:
        del sample_manifest["sections"]["identity"]
        errors = BindingManifest.validate(sample_manifest)
        assert any("identity" in e for e in errors)

    def test_section_not_dict(self, sample_manifest: dict[str, Any]) -> None:
        sample_manifest["sections"]["identity"] = "not a dict"
        errors = BindingManifest.validate(sample_manifest)
        assert any("must be a dict" in e for e in errors)

    def test_controls_not_list(self, sample_manifest: dict[str, Any]) -> None:
        sample_manifest["controls"] = "not a list"
        errors = BindingManifest.validate(sample_manifest)
        assert any("controls" in e and "list" in e for e in errors)

    def test_invalid_control_type(self, sample_manifest: dict[str, Any]) -> None:
        sample_manifest["controls"] = [{"type": "bad_type", "weight": 0.5}]
        errors = BindingManifest.validate(sample_manifest)
        assert any("unknown type" in e for e in errors)

    def test_invalid_control_weight(self, sample_manifest: dict[str, Any]) -> None:
        sample_manifest["controls"] = [{"type": "depth", "weight": 1.5}]
        errors = BindingManifest.validate(sample_manifest)
        assert any("weight" in e for e in errors)

    def test_blended_control_accepted(self, sample_manifest: dict[str, Any]) -> None:
        sample_manifest["controls"] = [
            {"type": "blended", "weight": 0.9, "children": []}
        ]
        errors = BindingManifest.validate(sample_manifest)
        assert errors == []

    def test_no_sections_key(self) -> None:
        m = BindingManifest.create(
            asset_type="prop",
            brief="test",
            identity={},
            geometry={},
            motion={},
            atmosphere={},
        )
        del m["sections"]
        errors = BindingManifest.validate(m)
        assert any("sections" in e for e in errors)


class TestValidateStrict:
    def test_raises_on_invalid(self) -> None:
        with pytest.raises(ManifestValidationError) as exc_info:
            BindingManifest.validate_strict({"version": "1.0", "asset_type": "nope"})
        assert len(exc_info.value.errors) > 0
        assert exc_info.value.code == "MANIFEST_INVALID"


class TestSerialisation:
    def test_to_json_and_back(
        self, sample_manifest: dict[str, Any], tmp_path: Path
    ) -> None:
        # Wrap in a BindingManifest for to_json
        obj = BindingManifest.__new__(BindingManifest)
        obj._data = sample_manifest

        path = tmp_path / "manifest.json"
        obj.to_json(path)

        loaded = BindingManifest.from_json(path)
        assert loaded.data["version"] == sample_manifest["version"]
        assert loaded.data["asset_type"] == sample_manifest["asset_type"]

    def test_from_json_not_found(self) -> None:
        with pytest.raises(ManifestNotFoundError):
            BindingManifest.from_json("/nonexistent/path.json")

    def test_repr(self, sample_manifest: dict[str, Any]) -> None:
        obj = BindingManifest.__new__(BindingManifest)
        obj._data = sample_manifest
        r = repr(obj)
        assert "BindingManifest" in r
        assert "character" in r


class TestGenerateControls:
    def test_basic_generation(self) -> None:
        controls = BindingManifest.generate_controls(
            ["depth", "canny"], [0.8, 0.6]
        )
        assert len(controls) == 2
        assert controls[0]["type"] == "depth"
        assert controls[0]["weight"] == 0.8
        assert controls[1]["type"] == "canny"

    def test_length_mismatch(self) -> None:
        with pytest.raises(ManifestValidationError):
            BindingManifest.generate_controls(["depth", "canny"], [0.8])

    def test_all_types(self) -> None:
        weights = [0.5] * len(ALL_CONTROL_TYPES)
        controls = BindingManifest.generate_controls(ALL_CONTROL_TYPES, weights)
        assert len(controls) == len(ALL_CONTROL_TYPES)
