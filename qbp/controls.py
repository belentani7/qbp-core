"""Control signal management for the Quadrachy Binding Protocol."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from numbers import Real
from dataclasses import dataclass, field, asdict
from typing import Any

from .errors import ControlSignalError, InvalidWeightError

# ── Canonical control types ────────────────────────────────────────────────────

DEPTH = "depth"
CANNY = "canny"
LINEART = "lineart"
NORMAL = "normal"
POSE = "pose"
IP_ADAPTER_STYLE = "ip_adapter_style"
FACE_LOCK = "face_lock"
REFERENCE = "reference"

ALL_CONTROL_TYPES: list[str] = [
    DEPTH, CANNY, LINEART, NORMAL, POSE, IP_ADAPTER_STYLE, FACE_LOCK, REFERENCE,
]


@dataclass(frozen=True, slots=True)
class ControlSignal:
    """Immutable value object representing a single control signal."""

    control_type: str
    weight: float
    source_image: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── factory ────────────────────────────────────────────────────────────

    @staticmethod
    def create(control_type: str, weight: float, **kwargs: Any) -> dict[str, Any]:
        """Create a validated control signal dict.

        Parameters
        ----------
        control_type:
            One of the canonical control types (e.g. ``DEPTH``, ``CANNY``).
        weight:
            Blending weight in ``[0, 1]``.
        **kwargs:
            Optional keys: ``source_image``, ``metadata``.

        Returns
        -------
        dict
            Serialisable control signal.
        """
        ControlSignal._validate_type(control_type)
        ControlSignal._validate_weight(weight)

        return {
            "type": control_type,
            "weight": round(weight, 4),
            "source_image": kwargs.get("source_image"),
            "metadata": kwargs.get("metadata", {}),
        }

    # ── validation ─────────────────────────────────────────────────────────

    @staticmethod
    def validate_weight(weight: float) -> bool:
        """Return ``True`` if *weight* is in the valid range ``[0, 1]``."""
        return isinstance(weight, Real) and not isinstance(weight, bool) and math.isfinite(float(weight)) and 0.0 <= weight <= 1.0

    @staticmethod
    def _validate_weight(weight: float) -> None:
        if not ControlSignal.validate_weight(weight):
            raise InvalidWeightError(weight)

    @staticmethod
    def _validate_type(control_type: str) -> None:
        if control_type not in ALL_CONTROL_TYPES:
            raise ControlSignalError(
                f"Unknown control type '{control_type}'. "
                f"Valid types: {', '.join(ALL_CONTROL_TYPES)}",
            )

    # ── blending ───────────────────────────────────────────────────────────

    @staticmethod
    def blend(signals: list[dict[str, Any]]) -> dict[str, Any]:
        """Blend multiple control signals into a single composite signal.

        The blended signal uses a weighted-average approach per type and
        records the constituent signals in ``children``.

        Parameters
        ----------
        signals:
            List of control-signal dicts (as produced by :meth:`create`).

        Returns
        -------
        dict
            A composite control signal with ``type="blended"``.
        """
        if not signals:
            raise ControlSignalError("Cannot blend an empty list of signals")

        # Only validated, finite, canonical controls can contribute.
        # This prevents negative, NaN and infinity weights from corrupting
        # normalized scores and avoids ambiguous composite signals.
        type_weights: dict[str, float] = {}
        total_weight = 0.0
        for sig in signals:
            if not isinstance(sig, dict):
                raise ControlSignalError("Each signal must be an object")
            ControlSignal._validate_type(sig.get("type"))
            if "weight" not in sig:
                raise ControlSignalError("Signal is missing its weight")
            ControlSignal._validate_weight(sig["weight"])
            t = sig["type"]
            w = float(sig["weight"])
            total_weight += w
            type_weights[t] = type_weights.get(t, 0.0) + w

        if total_weight == 0:
            raise ControlSignalError("Total blended weight must be > 0")

        blended = {
            "type": "blended",
            "weight": round(min(total_weight, 1.0), 4),
            "children": copy.deepcopy(signals),
            "type_weights": {k: round(v / total_weight, 4) for k, v in type_weights.items()},
            "signal_hash": _hash_signals(signals),
        }
        return blended


def _hash_signals(signals: list[dict[str, Any]]) -> str:
    """Deterministic hash of a list of control signals for change detection."""
    try:
        canonical = json.dumps(
            signals, sort_keys=True, separators=(",", ":"), allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ControlSignalError("Signals must contain JSON-serializable values") from exc
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
