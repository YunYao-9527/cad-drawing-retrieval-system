"""
Pairwise similarity visualization for engineering drawings.

This script produces:
1. A side-by-side attention-style similarity heatmap image.
2. A JSON report with standard similarity scores.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config_manager import get_config, init_config
from module.structure_features import build_structure_descriptor, compare_structure_descriptors
from services.cleaning_service import get_cleaning_service, init_cleaning_service
from services.feature_service import get_feature_service, init_feature_service
from services.ocr_service import compare_text_descriptors, get_ocr_service, init_ocr_service


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize pairwise similarity between two CAD drawings.")
    parser.add_argument("--query", required=True, help="Absolute path of query image")
    parser.add_argument("--target", required=True, help="Absolute path of target image")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "visualizations"),
        help="Directory to save visualization results",
    )
    parser.add_argument(
        "--method",
        choices=["patch_similarity", "grad_cam"],
        default="grad_cam",
        help="Visualization backend",
    )
    parser.add_argument("--use-cleaning", action="store_true", default=True, help="Enable cleaning before analysis")
    parser.add_argument("--use-ocr", action="store_true", default=True, help="Enable OCR text extraction")
    return parser.parse_args()


def init_services() -> None:
    init_config()
    config = get_config()
    if config.cleaning.enabled:
        model_path = config.cleaning.model_path
        if model_path and os.path.exists(model_path):
            init_cleaning_service(model_path)
    if config.innovation.enable_ocr_text_fusion:
        init_ocr_service()
    init_feature_service()


def empty_text_descriptor() -> Dict:
    return {
        "raw_text": "",
        "text_lines": [],
        "tokens": [],
        "number_tokens": [],
        "chargrams": [],
        "has_text": False,
        "source": "disabled",
    }


def prepare_image_bundle(image_path: str, use_cleaning: bool = True, use_ocr: bool = True) -> Dict:
    feature_service = get_feature_service()
    with open(image_path, "rb") as handle:
        image = Image.open(handle).convert("RGB")

    if use_cleaning:
        cleaned_image, mask_image, cleaning_meta = feature_service._apply_cleaning(image)
    else:
        cleaned_image = image.copy()
        mask_image = feature_service._fallback_keep_mask(cleaned_image)
        cleaning_meta = {"removed_regions": 0.0, "mask_coverage": 1.0}

    patch_mask = feature_service._build_patch_mask(mask_image)
    tensor = feature_service.preprocess(cleaned_image).unsqueeze(0).to(feature_service.device)
    with torch.no_grad():
        tokens = feature_service.model._encode_visual_tokens(tensor)
        embedding = feature_service.model(
            tensor,
            return_features=True,
            mask=patch_mask if feature_service.config.innovation.enable_masked_pooling else None,
        )
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

    patch_tokens = tokens[:, 1:, :]
    patch_tokens = patch_tokens / patch_tokens.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    patch_mask_np = (
        patch_mask.squeeze(0).detach().cpu().numpy().astype(np.float32)
        if patch_mask is not None
        else np.ones((patch_tokens.shape[1],), dtype=np.float32)
    )
    structure_descriptor = build_structure_descriptor(
        cleaned_image,
        mask_image=mask_image,
        grid_size=feature_service.config.innovation.descriptor_grid_size,
    )
    text_descriptor = get_ocr_service().extract_text_descriptor(image, mask_image) if use_ocr else empty_text_descriptor()

    patch_count = int(patch_tokens.shape[1])
    grid_size = int(math.sqrt(patch_count))
    if grid_size * grid_size != patch_count:
        raise ValueError(f"Patch token count is not a square grid: {patch_count}")

    return {
        "image_path": image_path,
        "original_image": image,
        "cleaned_image": cleaned_image,
        "mask_image": mask_image,
        "cleaning_meta": cleaning_meta,
        "embedding": embedding.squeeze(0).detach().cpu().numpy().astype(np.float32),
        "patch_tokens": patch_tokens.squeeze(0).detach().cpu().numpy().astype(np.float32),
        "patch_mask": patch_mask_np,
        "input_tensor": tensor.detach().cpu(),
        "patch_mask_tensor": patch_mask.detach().cpu() if patch_mask is not None else None,
        "grid_size": grid_size,
        "structure_descriptor": structure_descriptor,
        "text_descriptor": text_descriptor,
    }


def compute_ssim(image_a: Image.Image, image_b: Image.Image, size: int = 256) -> float:
    arr_a = np.asarray(image_a.convert("L").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32)
    arr_b = np.asarray(image_b.convert("L").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32)

    k1 = 0.01
    k2 = 0.03
    l = 255.0
    c1 = (k1 * l) ** 2
    c2 = (k2 * l) ** 2

    mu_a = cv2.GaussianBlur(arr_a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(arr_b, (11, 11), 1.5)

    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a_sq = cv2.GaussianBlur(arr_a * arr_a, (11, 11), 1.5) - mu_a_sq
    sigma_b_sq = cv2.GaussianBlur(arr_b * arr_b, (11, 11), 1.5) - mu_b_sq
    sigma_ab = cv2.GaussianBlur(arr_a * arr_b, (11, 11), 1.5) - mu_ab

    ssim_map = ((2 * mu_ab + c1) * (2 * sigma_ab + c2)) / (
        (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2) + 1e-6
    )
    return float(np.clip(ssim_map.mean(), 0.0, 1.0))


def _compute_patch_similarity_scores(query_bundle: Dict, target_bundle: Dict) -> Dict:
    query_tokens = query_bundle["patch_tokens"]
    target_tokens = target_bundle["patch_tokens"]
    sim_matrix = np.matmul(query_tokens, target_tokens.T)

    query_mask = query_bundle["patch_mask"] > 0.5
    target_mask = target_bundle["patch_mask"] > 0.5
    if not query_mask.all():
        sim_matrix[~query_mask, :] = 0.0
    if not target_mask.all():
        sim_matrix[:, ~target_mask] = 0.0

    query_topk = np.sort(sim_matrix, axis=1)[:, -3:]
    target_topk = np.sort(sim_matrix, axis=0)[-3:, :]
    query_patch_scores = query_topk.mean(axis=1)
    target_patch_scores = target_topk.mean(axis=0)

    embedding_cosine = float(
        np.dot(query_bundle["embedding"], target_bundle["embedding"])
        / (np.linalg.norm(query_bundle["embedding"]) * np.linalg.norm(target_bundle["embedding"]) + 1e-6)
    )
    semantic_similarity = float(np.clip((embedding_cosine + 1.0) / 2.0, 0.0, 1.0))
    structure_similarity = compare_structure_descriptors(
        query_bundle["structure_descriptor"],
        target_bundle["structure_descriptor"],
    )
    text_similarity = compare_text_descriptors(
        query_bundle["text_descriptor"],
        target_bundle["text_descriptor"],
    )
    patch_alignment = float(np.clip(query_patch_scores[query_mask].mean() if query_mask.any() else query_patch_scores.mean(), 0.0, 1.0))
    ssim_score = compute_ssim(query_bundle["cleaned_image"], target_bundle["cleaned_image"])

    return {
        "embedding_cosine": round(embedding_cosine, 6),
        "semantic_similarity": round(semantic_similarity, 6),
        "structure_similarity": round(float(structure_similarity), 6),
        "text_similarity": round(float(text_similarity), 6),
        "patch_alignment": round(patch_alignment, 6),
        "ssim": round(ssim_score, 6),
        "query_patch_scores": query_patch_scores.tolist(),
        "target_patch_scores": target_patch_scores.tolist(),
    }


def _forward_for_gradcam(
    image_tensor: torch.Tensor,
    patch_mask_tensor: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    feature_service = get_feature_service()
    model = feature_service.model

    tensor = image_tensor.to(feature_service.device).clone().detach().requires_grad_(True)
    patch_mask = patch_mask_tensor.to(feature_service.device) if patch_mask_tensor is not None else None

    tokens = model._encode_visual_tokens(tensor)
    if tokens is None:
        raise ValueError("Grad-CAM requires transformer patch tokens, but the current visual backbone did not expose them.")
    tokens.retain_grad()

    if model.adapter_type == "spatial_attention":
        logits = model.adapter(tokens, mask=patch_mask)
    else:
        pooled = tokens[:, 0, :]
        if patch_mask is not None and feature_service.config.innovation.enable_masked_pooling:
            pooled = model._masked_pool(tokens, patch_mask)
        logits = model.adapter(pooled)

    embedding = F.normalize(logits, p=2, dim=1)
    return embedding, tokens


def _gradcam_from_tokens(tokens: torch.Tensor, patch_mask: np.ndarray) -> np.ndarray:
    if tokens.grad is None:
        raise ValueError("Grad-CAM could not access token gradients.")

    patch_tokens = tokens[:, 1:, :]
    patch_grads = tokens.grad[:, 1:, :]
    channel_weights = patch_grads.mean(dim=1, keepdim=True)
    cam = torch.relu((patch_tokens * channel_weights).sum(dim=-1)).squeeze(0)
    cam_np = cam.detach().cpu().numpy().astype(np.float32)
    cam_np *= patch_mask.astype(np.float32)
    if cam_np.max() > cam_np.min():
        cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min())
    return cam_np


def compute_pair_scores(query_bundle: Dict, target_bundle: Dict, method: str = "grad_cam") -> Dict:
    base_scores = _compute_patch_similarity_scores(query_bundle, target_bundle)
    base_scores["visual_method"] = method

    if method == "patch_similarity":
        base_scores["query_visual_scores"] = list(base_scores["query_patch_scores"])
        base_scores["target_visual_scores"] = list(base_scores["target_patch_scores"])
        return base_scores

    feature_service = get_feature_service()
    feature_service.model.zero_grad(set_to_none=True)
    query_embedding, query_tokens = _forward_for_gradcam(
        query_bundle["input_tensor"],
        query_bundle.get("patch_mask_tensor"),
    )
    target_embedding, target_tokens = _forward_for_gradcam(
        target_bundle["input_tensor"],
        target_bundle.get("patch_mask_tensor"),
    )

    pair_score = F.cosine_similarity(query_embedding, target_embedding).sum()
    pair_score.backward()

    query_cam = _gradcam_from_tokens(query_tokens, query_bundle["patch_mask"])
    target_cam = _gradcam_from_tokens(target_tokens, target_bundle["patch_mask"])

    base_scores["gradcam_pair_score"] = round(float(pair_score.detach().cpu().item()), 6)
    base_scores["query_visual_scores"] = query_cam.tolist()
    base_scores["target_visual_scores"] = target_cam.tolist()
    return base_scores


def score_map_to_overlay(image: Image.Image, patch_scores: np.ndarray, grid_size: int) -> Image.Image:
    width, height = image.size
    score_map = patch_scores.reshape(grid_size, grid_size)
    normalized = score_map - score_map.min()
    if normalized.max() > 1e-6:
        normalized = normalized / normalized.max()
    normalized = (normalized * 255).astype(np.uint8)
    heatmap = cv2.resize(normalized, (width, height), interpolation=cv2.INTER_CUBIC)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    image_bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(image_bgr, 0.55, heatmap, 0.45, 0)
    return Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))


def build_canvas(query_bundle: Dict, target_bundle: Dict, scores: Dict) -> Image.Image:
    query_overlay = score_map_to_overlay(
        query_bundle["cleaned_image"],
        np.asarray(scores["query_visual_scores"], dtype=np.float32),
        query_bundle["grid_size"],
    )
    target_overlay = score_map_to_overlay(
        target_bundle["cleaned_image"],
        np.asarray(scores["target_visual_scores"], dtype=np.float32),
        target_bundle["grid_size"],
    )

    panel_size = (520, 360)
    panels = [
        query_bundle["original_image"].resize(panel_size, Image.Resampling.LANCZOS),
        query_overlay.resize(panel_size, Image.Resampling.LANCZOS),
        target_bundle["original_image"].resize(panel_size, Image.Resampling.LANCZOS),
        target_overlay.resize(panel_size, Image.Resampling.LANCZOS),
    ]

    canvas = Image.new("RGB", (panel_size[0] * 2 + 60, panel_size[1] * 2 + 220), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    draw.text((20, 18), "Pairwise Similarity Visualization", fill=(0, 0, 0))
    draw.text(
        (20, 48),
        (
            f"Method={scores['visual_method']} | "
            f"Semantic={scores['semantic_similarity']:.4f} | "
            f"Structure={scores['structure_similarity']:.4f} | "
            f"Text={scores['text_similarity']:.4f} | "
            f"Patch={scores['patch_alignment']:.4f} | "
            f"SSIM={scores['ssim']:.4f}"
        ),
        fill=(0, 0, 0),
    )

    labels = [
        "Query Original",
        f"Query {scores['visual_method']} Heatmap",
        "Target Original",
        f"Target {scores['visual_method']} Heatmap",
    ]
    positions = [(20, 100), (40 + panel_size[0], 100), (20, 130 + panel_size[1]), (40 + panel_size[0], 130 + panel_size[1])]
    for label, panel, (x, y) in zip(labels, panels, positions):
        draw.text((x, y - 22), label, fill=(0, 0, 0))
        canvas.paste(panel, (x, y))

    return canvas


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    init_services()

    query_bundle = prepare_image_bundle(args.query, use_cleaning=args.use_cleaning, use_ocr=args.use_ocr)
    target_bundle = prepare_image_bundle(args.target, use_cleaning=args.use_cleaning, use_ocr=args.use_ocr)
    scores = compute_pair_scores(query_bundle, target_bundle, method=args.method)
    canvas = build_canvas(query_bundle, target_bundle, scores)

    stem = f"{Path(args.query).stem}__VS__{Path(args.target).stem}"
    image_path = output_dir / f"{stem}.png"
    json_path = output_dir / f"{stem}.json"
    canvas.save(image_path)

    report = {
        "query": {
            "path": args.query,
            "ocr_text": query_bundle["text_descriptor"].get("raw_text", ""),
            "mask_coverage": query_bundle["cleaning_meta"].get("mask_coverage", 1.0),
        },
        "target": {
            "path": args.target,
            "ocr_text": target_bundle["text_descriptor"].get("raw_text", ""),
            "mask_coverage": target_bundle["cleaning_meta"].get("mask_coverage", 1.0),
        },
        "scores": {
            key: value
            for key, value in scores.items()
            if key not in {"query_patch_scores", "target_patch_scores", "query_visual_scores", "target_visual_scores"}
        },
    }
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved visualization: {image_path}")
    print(f"Saved report: {json_path}")


if __name__ == "__main__":
    main()
