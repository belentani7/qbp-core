"""Correction Patch generation and application."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .errors import PatchApplicationError
from .qa import QAReport


class CorrectionPatch:
    """Generate and apply correction patches to a binding manifest.

    A patch is a list of operations (``add``, ``remove``, ``modify``) derived
    from a QA report's failures.

    Example
    -------
    >>> patch = CorrectionPatch.from_report(qa_report, manifest_dict)
    >>> corrected = patch.apply(manifest_dict)
    """

    # ── construction ───────────────────────────────────────────────────────

    @classmethod
    def from_report(cls, qa_report: QAReport, manifest: dict[str, Any]) -> CorrectionPatch:
        """Build a correction patch from QA failures and the current manifest.

        Parameters
        ----------
        qa_report:
            Parsed QA report.
        manifest:
            Current binding manifest dict.

        Returns
        -------
        CorrectionPatch
            Ready to be applied via :meth:`apply`.
        """
        operations: list[dict[str, Any]] = []
        failures = qa_report.get_failures()

        for failure in failures:
            dimension = failure.get("dimension", "unknown")
            suggestion = failure.get("suggestion", "")
            details = failure.get("details", {})

            if dimension == "identity_fidelity":
                operations.extend(_identity_corrections(suggestion, details, manifest))
            elif dimension == "temporal_consistency":
                operations.extend(_temporal_corrections(suggestion, details, manifest))
            elif dimension == "pbr_validity":
                operations.extend(_pbr_corrections(suggestion, details, manifest))
            elif dimension == "aesthetic_coherence":
                operations.extend(_aesthetic_corrections(suggestion, details, manifest))
            else:
                # Generic correction
                operations.append({
                    "op": "modify",
                    "path": failure.get("path", ""),
                    "reason": suggestion or f"Fix {dimension} issue",
                    "value": details.get("recommended_value"),
                })

        return cls(operations=operations, source_report=qa_report)

    def __init__(
        self,
        operations: list[dict[str, Any]] | None = None,
        source_report: QAReport | None = None,
    ) -> None:
        self.operations = operations or []
        self.source_report = source_report

    # ── application ────────────────────────────────────────────────────────

    def apply(self, manifest: dict[str, Any]) -> dict[str, Any]:
        """Apply this patch to a manifest, returning a new dict.

        Parameters
        ----------
        manifest:
            The original binding manifest (not mutated).

        Returns
        -------
        dict
            Corrected manifest.
        """
        corrected = copy.deepcopy(manifest)

        for op in self.operations:
            op_type = op.get("op", "")
            path = op.get("path", "")

            try:
                if op_type == "modify":
                    _set_nested(corrected, path, op.get("value"))
                elif op_type == "add":
                    _add_nested(corrected, path, op.get("value"))
                elif op_type == "remove":
                    _remove_nested(corrected, path)
                elif op_type == "set_controls":
                    corrected["controls"] = op.get("value", [])
                else:
                    raise PatchApplicationError(f"Unknown operation: {op_type}")
            except (KeyError, IndexError, TypeError) as exc:
                raise PatchApplicationError(
                    f"Failed to apply op '{op_type}' at '{path}': {exc}"
                ) from exc

        # Update patch metadata only when operations were applied
        if self.operations:
            corrected.setdefault("_patch_history", []).append({
                "patch_ops": len(self.operations),
                "source_score": (
                    self.source_report.overall_score() if self.source_report else None
                ),
            })
        return corrected

    def __len__(self) -> int:
        return len(self.operations)

    def __repr__(self) -> str:
        return f"CorrectionPatch(operations={len(self.operations)})"


# ── Auto-correction loop ──────────────────────────────────────────────────────


def auto_correct(
    qa_report: QAReport,
    manifest: dict[str, Any],
    *,
    max_iterations: int = 3,
    validator: Any = None,
    reassess: Any = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply verified corrections without manufacturing improvement.

    An unchanged QA report is evidence about the ORIGINAL artifact only.
    Reapplying it creates duplicate controls/history, so iterative runs
    require a callable `reassess(candidate) -> QAReport` to obtain fresh
    measurements. Each candidate is validated before being committed.
    """
    if max_iterations < 0:
        raise ValueError("max_iterations must be non-negative")
    current = copy.deepcopy(manifest)
    applied_patches: list[dict[str, Any]] = []
    report = qa_report

    for iteration in range(max_iterations):
        if not report.needs_regeneration():
            break

        patch = CorrectionPatch.from_report(report, current)
        if not patch.operations:
            break

        candidate = patch.apply(current)
        # Ignore the audit entry when deciding whether the underlying asset
        # actually changed; stale suggestions must not manufacture progress.
        before = {k: v for k, v in current.items() if k != "_patch_history"}
        after = {k: v for k, v in candidate.items() if k != "_patch_history"}
        if before == after:
            break
        if validator is not None:
            errors = validator(candidate)
            if errors:
                break  # Keep the last valid manifest, do not return invalid work.

        if candidate == current:
            break
        current = candidate
        applied_patches.append({
            "iteration": iteration,
            "operations": copy.deepcopy(patch.operations),
            "pre_score": report.overall_score(),
        })
        if reassess is None:
            break  # Never reuse stale QA results to claim multiple fixes.
        report = reassess(current)
        if not isinstance(report, QAReport):
            raise TypeError("reassess must return a QAReport")

    return current, applied_patches


# ── Correction generators per dimension ───────────────────────────────────────


def _identity_corrections(
    suggestion: str, details: dict[str, Any], manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    """Generate ops to fix identity fidelity issues."""
    ops: list[dict[str, Any]] = []
    suggested = details.get("suggested_attributes", {})

    for attr, value in suggested.items():
        ops.append({
            "op": "modify",
            "path": f"sections.identity.{attr}",
            "reason": suggestion,
            "value": value,
        })

    # If no specific suggestions, add a flag to re-run identity check
    if not suggested:
        ops.append({
            "op": "modify",
            "path": "sections.identity._needs_review",
            "reason": suggestion,
            "value": True,
        })

    return ops


def _temporal_corrections(
    suggestion: str, details: dict[str, Any], manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    """Generate ops to fix temporal consistency issues."""
    ops: list[dict[str, Any]] = []

    # Tighten motion parameters
    motion_params = details.get("motion_params", {})
    for param, value in motion_params.items():
        ops.append({
            "op": "modify",
            "path": f"sections.motion.{param}",
            "reason": suggestion,
            "value": value,
        })

    # Add temporal smoothing control if not present
    controls = manifest.get("controls", [])
    has_temporal = any(c.get("type") == "reference" for c in controls)
    if not has_temporal and details.get("add_reference_control", False):
        ops.append({
            "op": "set_controls",
            "path": "controls",
            "reason": "Add reference control for temporal consistency",
            "value": controls + [{"type": "reference", "weight": 0.5}],
        })

    return ops


def _pbr_corrections(
    suggestion: str, details: dict[str, Any], manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    """Generate ops to fix PBR validity issues."""
    ops: list[dict[str, Any]] = []

    # Correct geometry parameters that affect PBR
    geometry_fixes = details.get("geometry_fixes", {})
    for param, value in geometry_fixes.items():
        ops.append({
            "op": "modify",
            "path": f"sections.geometry.{param}",
            "reason": suggestion,
            "value": value,
        })

    return ops


def _aesthetic_corrections(
    suggestion: str, details: dict[str, Any], manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    """Generate ops to fix aesthetic coherence issues."""
    ops: list[dict[str, Any]] = []

    atmosphere_fixes = details.get("atmosphere_fixes", {})
    for param, value in atmosphere_fixes.items():
        ops.append({
            "op": "modify",
            "path": f"sections.atmosphere.{param}",
            "reason": suggestion,
            "value": value,
        })

    # Add style control if aesthetic score is very low
    score = details.get("score", 1.0)
    if score < 0.4:
        controls = manifest.get("controls", [])
        ops.append({
            "op": "set_controls",
            "path": "controls",
            "reason": "Add style control for low aesthetic score",
            "value": controls + [{"type": "ip_adapter_style", "weight": 0.6}],
        })

    return ops


# ── Nested dict helpers ──────────────────────────────────────────────────────


def _set_nested(d: dict[str, Any], path: str, value: Any) -> None:
    """Set a value in a nested dict using a dot-separated path."""
    if not isinstance(path, str) or not path or any(not segment for segment in path.split(".")):
        raise PatchApplicationError(f"Invalid manifest path: {path!r}")
    keys = path.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
        if not isinstance(d, dict):
            raise PatchApplicationError(f"Cannot traverse non-object section: {key}")
    d[keys[-1]] = value


def _add_nested(d: dict[str, Any], path: str, value: Any) -> None:
    """Add a value, creating intermediate dicts as needed (same as set)."""
    _set_nested(d, path, value)


def _remove_nested(d: dict[str, Any], path: str) -> None:
    """Remove a value from a nested dict using a dot-separated path."""
    if not isinstance(path, str) or not path or any(not segment for segment in path.split(".")):
        raise PatchApplicationError(f"Invalid manifest path: {path!r}")
    keys = path.split(".")
    for key in keys[:-1]:
        if key not in d:
            raise KeyError(key)
        d = d[key]
    last = keys[-1]
    if last in d:
        del d[last]
