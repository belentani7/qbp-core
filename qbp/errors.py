"""Error codes and exceptions for the QBP core SDK."""

from __future__ import annotations


class QBPError(Exception):
    """Base exception for all QBP errors."""

    code: str = "QBP_ERROR"

    def __init__(self, message: str, *, detail: dict | None = None) -> None:
        super().__init__(message)
        self.detail = detail or {}


class ManifestValidationError(QBPError):
    """Raised when a binding manifest fails validation."""

    code = "MANIFEST_INVALID"

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(
            f"Manifest validation failed with {len(errors)} error(s)",
            detail={"errors": errors},
        )


class ManifestNotFoundError(QBPError):
    """Raised when a manifest file cannot be located."""

    code = "MANIFEST_NOT_FOUND"


class QAReportParseError(QBPError):
    """Raised when a QA report JSON cannot be parsed."""

    code = "QA_REPORT_PARSE_ERROR"


class PatchApplicationError(QBPError):
    """Raised when a correction patch cannot be applied."""

    code = "PATCH_APPLICATION_ERROR"


class ControlSignalError(QBPError):
    """Raised for invalid control signal parameters."""

    code = "CONTROL_SIGNAL_ERROR"


class ScoringError(QBPError):
    """Raised when a scoring function encounters bad input or a missing backend."""

    code = "SCORING_ERROR"


class InvalidWeightError(QBPError):
    """Raised when a control weight is outside [0, 1]."""

    code = "INVALID_WEIGHT"

    def __init__(self, weight: float) -> None:
        super().__init__(f"Weight must be in [0, 1], got {weight}")


# ── Error code registry ───────────────────────────────────────────────────────

ERROR_CODES: dict[str, type[QBPError]] = {
    exc.code: exc for exc in [
        ManifestValidationError,
        ManifestNotFoundError,
        QAReportParseError,
        PatchApplicationError,
        ControlSignalError,
        ScoringError,
        InvalidWeightError,
    ]
}


def error_from_code(code: str, message: str = "", **kwargs: object) -> QBPError:
    """Instantiate an exception from its string code."""
    cls = ERROR_CODES.get(code, QBPError)
    if cls is ManifestValidationError:
        return cls(errors=kwargs.get("errors", []))  # type: ignore[arg-type]
    if cls is InvalidWeightError:
        return cls(weight=kwargs.get("weight", 0.0))  # type: ignore[arg-type]
    return cls(message)
