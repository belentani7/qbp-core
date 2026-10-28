"""Public API for qbp-core.

The package root intentionally exposes the stable SDK surface documented in the
README so consumers do not need to depend on internal module paths.
"""

from .controls import (
    ALL_CONTROL_TYPES,
    CANNY,
    DEPTH,
    FACE_LOCK,
    IP_ADAPTER_STYLE,
    LINEART,
    NORMAL,
    POSE,
    REFERENCE,
    ControlSignal,
)
from .errors import (
    ControlSignalError,
    InvalidWeightError,
    ManifestNotFoundError,
    ManifestValidationError,
    PatchApplicationError,
    QAReportParseError,
    QBPError,
    ScoringError,
)
from .manifest import BindingManifest
from .patch import CorrectionPatch, auto_correct
from .qa import QAReport

__all__ = [
    "ALL_CONTROL_TYPES",
    "BindingManifest",
    "CANNY",
    "ControlSignal",
    "ControlSignalError",
    "CorrectionPatch",
    "DEPTH",
    "FACE_LOCK",
    "IP_ADAPTER_STYLE",
    "InvalidWeightError",
    "LINEART",
    "ManifestNotFoundError",
    "ManifestValidationError",
    "NORMAL",
    "PatchApplicationError",
    "POSE",
    "QBPError",
    "QAReport",
    "QAReportParseError",
    "REFERENCE",
    "ScoringError",
    "auto_correct",
]
