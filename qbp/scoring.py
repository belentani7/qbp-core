"""Consistency scoring across four QBP dimensions.

All scoring functions operate on plain data (file paths, dicts).  ML backends
(ArcFace, CLIP, LPIPS) are imported lazily so the package works without
``torch`` installed — missing backends raise :class:`ScoringError`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from .errors import ScoringError

# ── Backend protocols ────────────────────────────────────────────────────────


class _Backend(Protocol):
    """Minimal interface for a scoring backend."""
    def score(self, *args: Any, **kwargs: Any) -> float: ...


# ── Lazy ML backends ─────────────────────────────────────────────────────────

_arcface_model: Any = None
_clip_model: Any = None
_clip_preprocess: Any = None
_clip_tokenizer: Any = None
_lpips_fn: Any = None


def _get_arcface() -> Any:
    """Lazy-load ArcFace via facenet-pytorch."""
    global _arcface_model
    if _arcface_model is None:
        try:
            from facenet_pytorch import InceptionResnetV1
            import torch
        except ImportError as exc:
            raise ScoringError(
                "ArcFace requires 'facenet-pytorch' and 'torch'. "
                "Install with: pip install qbp-core[ml]"
            ) from exc
        _arcface_model = InceptionResnetV1(pretrained="vggface2").eval()
    return _arcface_model


def _get_clip() -> tuple[Any, Any, Any]:
    """Lazy-load CLIP model, preprocess, and tokenizer."""
    global _clip_model, _clip_preprocess, _clip_tokenizer
    if _clip_model is None:
        try:
            import clip
            import torch
        except ImportError as exc:
            raise ScoringError(
                "CLIP requires 'clip-anytorch' and 'torch'. "
                "Install with: pip install qbp-core[ml]"
            ) from exc
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=device)
        _clip_tokenizer = clip.tokenize
    return _clip_model, _clip_preprocess, _clip_tokenizer


def _get_lpips() -> Any:
    """Lazy-load LPIPS metric."""
    global _lpips_fn
    if _lpips_fn is None:
        try:
            import lpips
            import torch
        except ImportError as exc:
            raise ScoringError(
                "LPIPS requires 'lpips' and 'torch'. "
                "Install with: pip install qbp-core[ml]"
            ) from exc
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _lpips_fn = lpips.LPIPS(net="alex").to(device)
    return _lpips_fn


# ── Public scoring functions ─────────────────────────────────────────────────


def identity_score(image_a: str | Path, image_b: str | Path) -> float:
    """Compute identity similarity between two face images.

    Uses ArcFace (facenet-pytorch) for embedding extraction and cosine
    similarity.  Falls back to CLIP-based similarity when no face is detected.

    Parameters
    ----------
    image_a, image_b:
        Paths to the two images to compare.

    Returns
    -------
    float
        Similarity score in ``[0, 1]`` where 1 means identical identity.
    """
    try:
        return _identity_arcface(image_a, image_b)
    except ScoringError:
        # Fallback: CLIP perceptual similarity
        return _identity_clip_fallback(image_a, image_b)


def _identity_arcface(image_a: str | Path, image_b: str | Path) -> float:
    import torch
    from torchvision import transforms

    model = _get_arcface()
    device = next(model.parameters()).device

    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    img_a = transform(Image.open(image_a).convert("RGB")).unsqueeze(0).to(device)
    img_b = transform(Image.open(image_b).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        emb_a = model(img_a)
        emb_b = model(img_b)

    cosine_sim = torch.nn.functional.cosine_similarity(emb_a, emb_b).item()
    return float(max(0.0, min(1.0, (cosine_sim + 1) / 2)))


def _identity_clip_fallback(image_a: str | Path, image_b: str | Path) -> float:
    """CLIP-based perceptual similarity as identity fallback."""
    import torch

    model, preprocess, _ = _get_clip()
    device = next(model.parameters()).device

    img_a = preprocess(Image.open(image_a).convert("RGB")).unsqueeze(0).to(device)
    img_b = preprocess(Image.open(image_b).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        emb_a = model.encode_image(img_a)
        emb_b = model.encode_image(img_b)
        emb_a = emb_a / emb_a.norm(dim=-1, keepdim=True)
        emb_b = emb_b / emb_b.norm(dim=-1, keepdim=True)

    similarity = (emb_a @ emb_b.T).item()
    return float(max(0.0, min(1.0, (similarity + 1) / 2)))


def temporal_consistency(frames: list[str | Path] | list[Image.Image]) -> float:
    """Measure frame-to-frame temporal consistency using LPIPS.

    Parameters
    ----------
    frames:
        Ordered list of frame file paths or PIL Images.

    Returns
    -------
    float
        Mean consistency in ``[0, 1]`` where 1 means no perceptible flicker.
    """
    if len(frames) < 2:
        return 1.0

    try:
        return _temporal_lpips(frames)
    except ScoringError:
        return _temporal_pixel_fallback(frames)


def _temporal_lpips(frames: list[str | Path] | list[Image.Image]) -> float:
    import torch

    lpips_fn = _get_lpips()
    device = next(lpips_fn.parameters()).device

    differences: list[float] = []
    prev_tensor: torch.Tensor | None = None

    for frame in frames:
        img = _load_image_tensor(frame, device)
        if prev_tensor is not None:
            with torch.no_grad():
                diff = lpips_fn(prev_tensor, img).item()
            differences.append(diff)
        prev_tensor = img

    if not differences:
        return 1.0

    # LPIPS distance: 0 = identical, ~1 = very different.  Invert for consistency.
    mean_diff = float(np.mean(differences))
    return round(max(0.0, 1.0 - mean_diff), 4)


def _temporal_pixel_fallback(frames: list[str | Path] | list[Image.Image]) -> float:
    """Lightweight pixel-difference fallback when torch is not available."""
    prev_arr: np.ndarray | None = None
    differences: list[float] = []

    for frame in frames:
        img = _load_image_pil(frame)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if prev_arr is not None and arr.shape == prev_arr.shape:
            diff = float(np.mean(np.abs(arr - prev_arr)))
            differences.append(diff)
        prev_arr = arr

    if not differences:
        return 1.0
    mean_diff = float(np.mean(differences))
    return round(max(0.0, 1.0 - mean_diff), 4)


def pbr_validity(texture_set: dict[str, str | Path] | list[str | Path]) -> float:
    """Evaluate PBR texture set completeness and validity.

    Parameters
    ----------
    texture_set:
        Either a dict mapping channel names to file paths (e.g.
        ``{"albedo": "...", "normal": "...", "roughness": "..."}``) or a list
        of file paths that follow the naming convention.

    Returns
    -------
    float
        Validity score in ``[0, 1]``.
    """
    required_channels = {"albedo", "normal", "roughness", "metallic"}
    optional_channels = {"ao", "emission", "height", "displacement"}

    if isinstance(texture_set, dict):
        channels = set(texture_set.keys())
        paths = texture_set
    else:
        # Infer channels from filenames
        channels = set()
        paths = {}
        for p in texture_set:
            p_str = str(p).lower()
            for ch in required_channels | optional_channels:
                if ch in p_str:
                    channels.add(ch)
                    paths[ch] = p

    # Score based on required channel coverage
    present_required = channels & required_channels
    required_score = len(present_required) / len(required_channels) if required_channels else 1.0

    # Bonus for optional channels
    present_optional = channels & optional_channels
    optional_bonus = len(present_optional) / len(optional_channels) * 0.2 if optional_channels else 0.0

    # Validate that referenced files exist (if paths are provided)
    file_score = 1.0
    if isinstance(paths, dict):
        existing = sum(1 for p in paths.values() if Path(p).exists())
        if paths:
            file_score = existing / len(paths)

    return round(min(1.0, required_score * 0.8 + optional_bonus + file_score * 0.2), 4)


def overall_consistency(scores: dict[str, float]) -> float:
    """Compute overall consistency from per-dimension scores.

    Parameters
    ----------
    scores:
        Dict with keys matching ``DEFAULT_DIMENSION_WEIGHTS`` and values in
        ``[0, 1]``.

    Returns
    -------
    float
        Weighted average in ``[0, 1]``.
    """
    from .qa import DEFAULT_DIMENSION_WEIGHTS

    total_weight = 0.0
    weighted_sum = 0.0

    for dim, weight in DEFAULT_DIMENSION_WEIGHTS.items():
        if dim in scores:
            weighted_sum += scores[dim] * weight
            total_weight += weight

    if total_weight == 0:
        return 0.0

    return round(weighted_sum / total_weight, 4)


# ── Image loading helpers ────────────────────────────────────────────────────


def _load_image_tensor(path: str | Path, device: str) -> Any:
    """Load an image and convert to a normalised tensor."""
    import torch
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    img = Image.open(path).convert("RGB") if isinstance(path, (str, Path)) else path
    return transform(img).unsqueeze(0).to(device)


def _load_image_pil(source: str | Path | Image.Image) -> Image.Image:
    """Load an image from path or return PIL Image as-is."""
    if isinstance(source, Image.Image):
        return source
    return Image.open(source).convert("RGB")
