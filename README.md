# qbp-core

Core SDK for the **Quadrachy Binding Protocol** — a structured format for defining, validating, and quality-checking AI-generated assets across four consistency dimensions: identity fidelity, temporal consistency, PBR validity, and aesthetic coherence.

## Installation

Install from source (not yet published to PyPI):

```bash
git clone https://github.com/belentani7/qbp-core.git
cd qbp-core
pip install -e .

# With ML scoring backends (ArcFace, CLIP, LPIPS):
pip install -e ".[ml]"

# Development:
pip install -e ".[dev]"
```

**Requires Python >= 3.10.**

## Quick Start

```python
from qbp.manifest import BindingManifest
from qbp.qa import QAReport
from qbp.patch import CorrectionPatch, auto_correct
from qbp.controls import ControlSignal

# 1. Create a binding manifest
manifest = BindingManifest.create(
    asset_type="character",
    brief="A cyberpunk samurai with glowing katana",
    identity={"name": "Ryu", "age": 28, "origin": "Neo-Tokyo"},
    geometry={"height": 180, "build": "athletic", "wingspan": 175},
    motion={"idle": "ready_stance", "walk": "predatory"},
    atmosphere={"mood": "tense", "lighting": "neon_rim"},
    tags=["cyberpunk", "warrior"],
)

# 2. Validate
errors = BindingManifest.validate(manifest)
assert not errors  # empty list = valid

# 3. Generate control signals
controls = BindingManifest.generate_controls(
    control_types=["depth", "canny", "ip_adapter_style"],
    weights=[0.8, 0.6, 0.5],
)
manifest["controls"] = controls

# 4. Serialise
BindingManifest.validate_strict(manifest)  # raises ManifestValidationError on failure
manifest_obj = BindingManifest.create(**manifest)  # wrap for file I/O
manifest_obj.to_json("output/manifest.json")
loaded = BindingManifest.from_json("output/manifest.json")

# 5. QA check (from a report dict)
report = QAReport.parse({
    "checks": [
        {"dimension": "identity_fidelity", "passed": True, "score": 0.95},
        {"dimension": "temporal_consistency", "passed": False, "score": 0.42,
         "suggestion": "Add reference control", "path": "controls"},
    ],
    "dimensions": {
        "identity_fidelity": {"score": 0.95},
        "temporal_consistency": {"score": 0.42},
        "pbr_validity": {"score": 0.88},
    },
})
print(report.summary())
# QA Report [unknown]: score=0.74, checks=1/2 passed, regen=no

# 6. Auto-correct
corrected, patches = auto_correct(report, manifest, max_iterations=3)
```

## API Reference

### `qbp.manifest.BindingManifest`

| Method | Description |
|---|---|
| `create(asset_type, brief, identity, geometry, motion, atmosphere, *, tags, controls)` | Build a manifest dict |
| `validate(manifest)` | Returns `list[str]` of errors (empty = valid) |
| `validate_strict(manifest)` | Raises `ManifestValidationError` on failure |
| `from_json(path)` | Load manifest from JSON file |
| `to_json(path)` | Write manifest to JSON file |
| `generate_controls(types, weights)` | Generate control signal dicts |

### `qbp.qa.QAReport`

| Method | Description |
|---|---|
| `parse(source)` | Parse from dict, JSON string, or file path |
| `overall_score()` | Weighted score in `[0, 1]` |
| `needs_regeneration(threshold=0.7)` | True if score < threshold |
| `get_failures()` | List of failed check dicts |
| `summary()` | Human-readable one-liner |

### `qbp.patch.CorrectionPatch`

| Method | Description |
|---|---|
| `from_report(qa_report, manifest)` | Build patch from QA failures |
| `apply(manifest)` | Apply patch, returns new manifest dict |
| `auto_correct(report, manifest, max_iterations=3)` | Iterative correction loop |

### `qbp.controls.ControlSignal`

| Method | Description |
|---|---|
| `create(type, weight, **kwargs)` | Create a validated control signal |
| `validate_weight(weight)` | Check weight is in `[0, 1]` |
| `blend(signals)` | Blend multiple signals into one composite |

**Canonical control types:** `DEPTH`, `CANNY`, `LINEART`, `NORMAL`, `POSE`, `IP_ADAPTER_STYLE`, `FACE_LOCK`, `REFERENCE`

### `qbp.scoring` (requires `qbp-core[ml]`)

| Function | Description |
|---|---|
| `identity_score(img_a, img_b)` | Face identity similarity via ArcFace |
| `temporal_consistency(frames)` | Frame-to-frame consistency via LPIPS |
| `pbr_validity(texture_set)` | PBR texture set completeness |
| `overall_consistency(scores)` | Weighted 4-dimension average |

### Error Hierarchy

```
QBPError
├── ManifestValidationError
├── ManifestNotFoundError
├── QAReportParseError
├── PatchApplicationError
├── ControlSignalError
├── ScoringError
└── InvalidWeightError
```

All errors carry a `.code` string and optional `.detail` dict.

## Development

```bash
git clone https://github.com/belentani7/qbp-core.git
cd qbp-core
pip install -e ".[dev]"
pytest
```

## License

MIT
