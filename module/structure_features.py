"""
Utilities for lightweight structure-aware descriptors.

These descriptors are used for re-ranking retrieved engineering drawings
after the CLIP/Qdrant candidate search stage.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
from PIL import Image


def _as_float_array(values: Optional[Iterable[float]]) -> np.ndarray:
    if values is None:
        return np.zeros(0, dtype=np.float32)
    array = np.asarray(list(values), dtype=np.float32)
    return array


def _to_foreground_map(image: Image.Image, threshold: int = 245) -> np.ndarray:
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    return (gray < threshold).astype(np.float32)


def build_structure_descriptor(
    image: Image.Image,
    mask_image: Optional[Image.Image] = None,
    grid_size: int = 16,
) -> Dict[str, List[float] | float]:
    """Build a compact geometric occupancy descriptor from a cleaned drawing."""
    foreground = _to_foreground_map(image)

    if mask_image is not None:
        mask = np.asarray(mask_image.convert("L"), dtype=np.float32) / 255.0
        if mask.shape != foreground.shape:
            mask = np.asarray(
                mask_image.convert("L").resize(image.size, Image.Resampling.NEAREST),
                dtype=np.float32,
            ) / 255.0
        foreground = foreground * (mask > 0.5).astype(np.float32)

    foreground_ratio = float(foreground.mean())

    occupancy = Image.fromarray((foreground * 255).astype(np.uint8)).resize(
        (grid_size, grid_size),
        Image.Resampling.BILINEAR,
    )
    occupancy_array = np.asarray(occupancy, dtype=np.float32) / 255.0

    horizontal_profile = occupancy_array.mean(axis=1)
    vertical_profile = occupancy_array.mean(axis=0)

    return {
        "grid": occupancy_array.flatten().round(6).tolist(),
        "horizontal_profile": horizontal_profile.round(6).tolist(),
        "vertical_profile": vertical_profile.round(6).tolist(),
        "foreground_ratio": round(foreground_ratio, 6),
    }


def compare_structure_descriptors(
    query_descriptor: Optional[Dict[str, List[float] | float]],
    target_descriptor: Optional[Dict[str, List[float] | float]],
) -> float:
    """Compare two lightweight structure descriptors and return [0, 1] score."""
    if not query_descriptor or not target_descriptor:
        return 0.0

    query_grid = _as_float_array(query_descriptor.get("grid"))
    target_grid = _as_float_array(target_descriptor.get("grid"))
    if query_grid.size == 0 or target_grid.size == 0 or query_grid.size != target_grid.size:
        return 0.0

    def safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    query_h = _as_float_array(query_descriptor.get("horizontal_profile"))
    target_h = _as_float_array(target_descriptor.get("horizontal_profile"))
    query_v = _as_float_array(query_descriptor.get("vertical_profile"))
    target_v = _as_float_array(target_descriptor.get("vertical_profile"))

    grid_similarity = safe_cosine(query_grid, target_grid)
    horizontal_similarity = safe_cosine(query_h, target_h)
    vertical_similarity = safe_cosine(query_v, target_v)

    query_ratio = float(query_descriptor.get("foreground_ratio", 0.0))
    target_ratio = float(target_descriptor.get("foreground_ratio", 0.0))
    density_similarity = max(0.0, 1.0 - abs(query_ratio - target_ratio))

    final_score = (
        0.55 * grid_similarity
        + 0.15 * horizontal_similarity
        + 0.15 * vertical_similarity
        + 0.15 * density_similarity
    )
    return float(max(0.0, min(1.0, final_score)))
