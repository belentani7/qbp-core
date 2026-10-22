"""Regression tests for the package-root public API."""

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
