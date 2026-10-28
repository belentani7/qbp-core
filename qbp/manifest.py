"""Binding Manifest creation, validation, and serialisation."""

from __future__ import annotations

import copy
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .controls import ALL_CONTROL_TYPES, ControlSignal
from .errors import ManifestNotFoundError, ManifestValidationError

# ── Schema constants ──────────────────────────────────────────────────────────

MANIFEST_VERSION = "1.0"
REQUIRED_ASSET_TYPES = {"character", "prop", "environment", "vfx", "ui"}
REQUIRED_SECTIONS = ("identity", "geometry", "motion", "atmosphere")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class BindingManifest:
    """Create, validate, and serialise a QBP binding manifest.

    Example
    -------
    >>> manifest = BindingManifest.create(
    ...     asset_type="character",
    ...     brief="A cyberpunk samurai",
    ...     identity={"name": "Ryu", "age": 28},
    ...     geometry={"height": 180, "build": "athletic"},
    ...     motion={"idle": "ready_stance"},
    ...     atmosphere={"mood": "tense"},
    ... )
    >>> errors = BindingManifest.validate(manifest)
    >>> assert not errors
    """

    # ── creation ───────────────────────────────────────────────────────────

    @staticmethod
    def create(
        asset_type: str,
        brief: str,
        identity: dict[str, Any],
        geometry: dict[str, Any],
        motion: dict[str, Any],
        atmosphere: dict[str, Any],
        *,
        tags: list[str] | None = None,
        controls: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build a binding manifest dict.

        Parameters
        ----------
        asset_type:
            One of ``REQUIRED_ASSET_TYPES``.
        brief:
            Free-text description of the asset (1-2 000 chars).
        identity, geometry, motion, atmosphere:
            Required section dicts.
        tags:
            Optional descriptive tags.
        controls:
            Optional pre-built control signals.

        Returns
        -------
        dict
            A complete binding manifest.
        """
        manifest: dict[str, Any] = {
            "version": MANIFEST_VERSION,
            "id": str(uuid.uuid4()),
            "created_at": _now_iso(),
            "asset_type": asset_type,
            "brief": brief,
            "tags": tags or [],
            "sections": {
                "identity": identity,
                "geometry": geometry,
                "motion": motion,
                "atmosphere": atmosphere,
            },
            "controls": controls or [],
        }
        return manifest

    # ── validation ─────────────────────────────────────────────────────────

    @staticmethod
    def validate(manifest: Any) -> list[str]:
        """Validate a manifest dict and return a list of error strings.

        An empty list means the manifest is valid.
        """
        errors: list[str] = []
        if not isinstance(manifest, dict):
            return ["Manifest must be a dict"]

        # version
        if manifest.get("version") != MANIFEST_VERSION:
            errors.append(f"Unsupported manifest version: {manifest.get('version')!r}")

        # asset_type
        at = manifest.get("asset_type", "")
        if not isinstance(at, str) or at not in REQUIRED_ASSET_TYPES:
            errors.append(
                f"Invalid asset_type '{at}'. Must be one of: {', '.join(sorted(REQUIRED_ASSET_TYPES))}"
            )

        # brief
        brief = manifest.get("brief", "")
        if not isinstance(brief, str) or len(brief) < 1:
            errors.append("Brief must be a non-empty string")
        elif len(brief) > 2000:
            errors.append(f"Brief exceeds 2000 characters (got {len(brief)})")

        # sections
        if "sections" not in manifest:
            errors.append("Missing required 'sections' key")
        sections = manifest.get("sections", {})
        if not isinstance(sections, dict):
            errors.append("'sections' must be a dict")
        else:
            for section in REQUIRED_SECTIONS:
                if section not in sections:
                    errors.append(f"Missing required section: '{section}'")
                elif not isinstance(sections[section], dict):
                    errors.append(f"Section '{section}' must be a dict")

        # controls (optional, but validate if present)
        controls = manifest.get("controls", [])
        if not isinstance(controls, list):
            errors.append("'controls' must be a list")
        else:
            for i, ctrl in enumerate(controls):
                if not isinstance(ctrl, dict):
                    errors.append(f"controls[{i}] must be a dict")
                    continue
                ctrl_type = ctrl.get("type", "")
                if ctrl_type not in ALL_CONTROL_TYPES and ctrl_type != "blended":
                    errors.append(
                        f"controls[{i}] has unknown type '{ctrl_type}'"
                    )
                weight = ctrl.get("weight", -1)
                if not isinstance(weight, (int, float)) or not ControlSignal.validate_weight(weight):
                    errors.append(
                        f"controls[{i}] has invalid weight: {weight!r}"
                    )

        return errors

    @classmethod
    def validate_strict(cls, manifest: dict[str, Any]) -> None:
        """Raise :class:`ManifestValidationError` if the manifest is invalid."""
        errors = cls.validate(manifest)
        if errors:
            raise ManifestValidationError(errors)

    # ── serialisation ──────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, manifest: dict[str, Any]) -> BindingManifest:
        """Wrap a validated manifest for file I/O without sharing mutable state."""
        cls.validate_strict(manifest)
        instance = cls.__new__(cls)
        instance._data = copy.deepcopy(manifest)
        return instance

    @classmethod
    def from_json(cls, path: str | Path) -> BindingManifest:
        """Load a manifest from a JSON file.

        Returns a :class:`BindingManifest` whose underlying data is the parsed
        dict.  The class methods :meth:`validate` etc. all accept the raw dict,
        so use ``manifest.data`` to access it.
        """
        p = Path(path)
        if not p.exists():
            raise ManifestNotFoundError(f"Manifest not found: {p}")
        with p.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        instance = cls.__new__(cls)
        instance._data = data  # type: ignore[attr-defined]
        return instance

    @property
    def data(self) -> dict[str, Any]:
        """Access the underlying manifest dict."""
        return self._data  # type: ignore[attr-defined]

    def to_json(self, path: str | Path) -> None:
        """Write the manifest to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            json.dump(self._data, fh, indent=2, ensure_ascii=False)

    # ── control generation ─────────────────────────────────────────────────

    @staticmethod
    def generate_controls(
        control_types: list[str],
        weights: list[float],
    ) -> list[dict[str, Any]]:
        """Generate a list of control signal dicts from types and weights.

        Parameters
        ----------
        control_types:
            List of canonical control type strings.
        weights:
            Parallel list of weights (same length as *control_types*).

        Returns
        -------
        list[dict]
        """
        if len(control_types) != len(weights):
            raise ManifestValidationError(
                [f"control_types ({len(control_types)}) and weights ({len(weights)}) must have equal length"]
            )
        return [
            ControlSignal.create(ct, w)
            for ct, w in zip(control_types, weights)
        ]

    def __repr__(self) -> str:
        data = getattr(self, "_data", {})
        return (
            f"BindingManifest(id={data.get('id', '?')!r}, "
            f"asset_type={data.get('asset_type', '?')!r})"
        )
