"""Unit tests for qbp.controls."""

from __future__ import annotations

import pytest

from qbp.controls import (
    ControlSignal,
    ALL_CONTROL_TYPES,
    DEPTH, CANNY, LINEART, NORMAL, POSE,
    IP_ADAPTER_STYLE, FACE_LOCK, REFERENCE,
)
from qbp.errors import ControlSignalError, InvalidWeightError


class TestConstants:
    def test_all_types_is_list(self) -> None:
        assert isinstance(ALL_CONTROL_TYPES, list)
        assert len(ALL_CONTROL_TYPES) == 8

    def test_type_constants_match(self) -> None:
        assert DEPTH == "depth"
        assert CANNY == "canny"
        assert LINEART == "lineart"
        assert NORMAL == "normal"
        assert POSE == "pose"
        assert IP_ADAPTER_STYLE == "ip_adapter_style"
        assert FACE_LOCK == "face_lock"
        assert REFERENCE == "reference"

    def test_all_types_contain_constants(self) -> None:
        for ct in [DEPTH, CANNY, LINEART, NORMAL, POSE, IP_ADAPTER_STYLE, FACE_LOCK, REFERENCE]:
            assert ct in ALL_CONTROL_TYPES


class TestCreate:
    def test_basic_creation(self) -> None:
        sig = ControlSignal.create("depth", 0.8)
        assert sig["type"] == "depth"
        assert sig["weight"] == 0.8
        assert sig["source_image"] is None
        assert sig["metadata"] == {}

    def test_with_optional_fields(self) -> None:
        sig = ControlSignal.create(
            "canny", 0.6,
            source_image="/path/to/img.png",
            metadata={"sigma": 100},
        )
        assert sig["source_image"] == "/path/to/img.png"
        assert sig["metadata"]["sigma"] == 100

    def test_weight_rounded(self) -> None:
        sig = ControlSignal.create("depth", 0.123456789)
        assert sig["weight"] == 0.1235

    def test_weight_zero_valid(self) -> None:
        sig = ControlSignal.create("depth", 0.0)
        assert sig["weight"] == 0.0

    def test_weight_one_valid(self) -> None:
        sig = ControlSignal.create("depth", 1.0)
        assert sig["weight"] == 1.0

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(ControlSignalError, match="Unknown control type"):
            ControlSignal.create("invalid", 0.5)

    def test_weight_above_one_raises(self) -> None:
        with pytest.raises(InvalidWeightError):
            ControlSignal.create("depth", 1.5)

    def test_weight_below_zero_raises(self) -> None:
        with pytest.raises(InvalidWeightError):
            ControlSignal.create("depth", -0.1)

    def test_all_valid_types(self) -> None:
        for ct in ALL_CONTROL_TYPES:
            sig = ControlSignal.create(ct, 0.5)
            assert sig["type"] == ct


class TestValidateWeight:
    @pytest.mark.parametrize("w", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_valid_weights(self, w: float) -> None:
        assert ControlSignal.validate_weight(w) is True

    @pytest.mark.parametrize("w", [-0.1, -1.0, 1.1, 2.0, 100.0])
    def test_invalid_weights(self, w: float) -> None:
        assert ControlSignal.validate_weight(w) is False

    def test_boundary_values(self) -> None:
        assert ControlSignal.validate_weight(0.0) is True
        assert ControlSignal.validate_weight(1.0) is True
        assert ControlSignal.validate_weight(-0.0001) is False
        assert ControlSignal.validate_weight(1.0001) is False


class TestBlend:
    def test_basic_blend(self) -> None:
        signals = [
            {"type": "depth", "weight": 0.8},
            {"type": "canny", "weight": 0.6},
        ]
        blended = ControlSignal.blend(signals)
        assert blended["type"] == "blended"
        assert blended["weight"] == 1.0
        assert "children" in blended
        assert len(blended["children"]) == 2
        assert "type_weights" in blended
        assert "signal_hash" in blended

    def test_blend_normalised_weights(self) -> None:
        signals = [
            {"type": "depth", "weight": 0.8},
            {"type": "canny", "weight": 0.2},
        ]
        blended = ControlSignal.blend(signals)
        assert blended["type_weights"]["depth"] == 0.8
        assert blended["type_weights"]["canny"] == 0.2

    def test_blend_same_type(self) -> None:
        signals = [
            {"type": "depth", "weight": 0.3},
            {"type": "depth", "weight": 0.7},
        ]
        blended = ControlSignal.blend(signals)
        assert blended["type_weights"]["depth"] == 1.0

    def test_blend_empty_raises(self) -> None:
        with pytest.raises(ControlSignalError, match="empty"):
            ControlSignal.blend([])

    def test_blend_zero_total_weight_raises(self) -> None:
        signals = [
            {"type": "depth", "weight": 0.0},
            {"type": "canny", "weight": 0.0},
        ]
        with pytest.raises(ControlSignalError, match="Total blended weight"):
            ControlSignal.blend(signals)

    def test_blend_weight_capped_at_one(self) -> None:
        signals = [
            {"type": "depth", "weight": 0.8},
            {"type": "canny", "weight": 0.8},
        ]
        blended = ControlSignal.blend(signals)
        assert blended["weight"] == 1.0  # min(1.6, 1.0)

    def test_signal_hash_deterministic(self) -> None:
        signals = [{"type": "depth", "weight": 0.5}]
        h1 = ControlSignal.blend(signals)["signal_hash"]
        h2 = ControlSignal.blend(signals)["signal_hash"]
        assert h1 == h2

    def test_signal_hash_different_for_different_signals(self) -> None:
        h1 = ControlSignal.blend([{"type": "depth", "weight": 0.5}])["signal_hash"]
        h2 = ControlSignal.blend([{"type": "canny", "weight": 0.5}])["signal_hash"]
        assert h1 != h2

    def test_blend_single_signal(self) -> None:
        signals = [{"type": "pose", "weight": 0.9}]
        blended = ControlSignal.blend(signals)
        assert blended["type"] == "blended"
        assert blended["weight"] == 0.9
        assert blended["type_weights"]["pose"] == 1.0



@pytest.mark.parametrize("weight", [float("nan"), float("inf"), -float("inf"), True, "0.5", None])
def test_create_rejects_nonfinite_or_non_numeric_weight(weight):
    assert ControlSignal.validate_weight(weight) is False
    with pytest.raises(InvalidWeightError):
        ControlSignal.create("depth", weight)


@pytest.mark.parametrize("weight", [float("nan"), float("inf"), -float("inf"), -0.1, 1.1, True, "0.8"])
def test_blend_rejects_invalid_child_weights(weight):
    with pytest.raises(InvalidWeightError):
        ControlSignal.blend([{"type": "depth", "weight": weight}])


def test_blend_rejects_noncanonical_types_or_missing_weights():
    with pytest.raises(ControlSignalError, match="Unknown control type"):
        ControlSignal.blend([{"type": "not-a-control", "weight": 0.5}])
    with pytest.raises(ControlSignalError, match="missing its weight"):
        ControlSignal.blend([{"type": "depth"}])
    with pytest.raises(ControlSignalError, match="object"):
        ControlSignal.blend(["depth"])


def test_blend_copies_nested_metadata_instead_of_aliasing_caller():
    signals = [{"type": "depth", "weight": 0.5, "metadata": {"tags": ["original"]}}]
    result = ControlSignal.blend(signals)
    signals[0]["metadata"]["tags"].append("later")
    assert result["children"][0]["metadata"]["tags"] == ["original"]


def test_blend_rejects_nonjson_signal_metadata():
    with pytest.raises(ControlSignalError, match="JSON-serializable"):
        ControlSignal.blend([{"type": "depth", "weight": 0.5, "metadata": {"value": float("nan")}}])
