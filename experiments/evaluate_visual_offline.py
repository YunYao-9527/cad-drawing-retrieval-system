"""
Offline visual retrieval evaluation for thesis-grade ablations.

Why this script exists:
1. The online API evaluator only switches query-time flags, while the gallery
   index may still come from a different preprocessing mode.
2. For rigorous ablations we need gallery and query features to be computed
   under the same mode.
3. This script caches mode-specific gallery features locally and evaluates
   retrieval without depending on Qdrant or the FastAPI service.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config_manager import get_config, init_config
from database.vector_db import get_vector_db, init_vector_db
from module.structure_features import compare_structure_descriptors
from services.feature_service import get_feature_service, init_feature_service


VISUAL_MODES = {
    "baseline": {
        "label": "Baseline",
        "use_cleaning": False,
        "use_masked_pooling": False,
        "enable_structure_rerank": False,
    },
    "cleaning_only": {
        "label": "YOLO Cleaning Only",
        "use_cleaning": True,
        "use_masked_pooling": False,
        "enable_structure_rerank": False,
    },
    "masked_pooling": {
        "label": "YOLO Cleaning + Masked Visual",
        "use_cleaning": True,
        "use_masked_pooling": True,
        "enable_structure_rerank": False,
    },
    "full_model": {
        "label": "YOLO Cleaning + Structure",
        "use_cleaning": True,
        "use_masked_pooling": True,
        "enable_structure_rerank": True,
    },
    "attention_visual": {
        "label": "Attention Visual (No Cleaning)",
        "use_cleaning": False,
        "use_masked_pooling": True,
        "enable_structure_rerank": False,
    },
    "attention_structure": {
        "label": "Attention + Structure (No Cleaning)",
        "use_cleaning": False,
        "use_masked_pooling": True,
        "enable_structure_rerank": True,
    },
}


@dataclass(frozen=True)
class GalleryItem:
    index: int
    filepath: str
    rel_path: str
    filename: str
    image_class: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline visual retrieval evaluation with mode-aligned gallery/query features.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=32, help="Feature extraction batch size.")
    parser.add_argument("--top-k", type=int, default=10, help="Summary metric depth.")
    parser.add_argument("--metric-depth", type=int, default=0, help="Retrieval depth. Use 0 for auto.")
    parser.add_argument("--max-queries-per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--modes",
        default="baseline,cleaning_only,masked_pooling,full_model,attention_visual,attention_structure",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "results_visual_offline"),
    )
    parser.add_argument(
        "--cache-dir",
        default=str(Path(__file__).resolve().parent / "cache_visual"),
    )
    parser.add_argument("--gallery-limit", type=int, default=0, help="Debug only. Use 0 for full gallery.")
    parser.add_argument("--rebuild-cache", action="store_true", help="Ignore any existing cached mode features.")
    return parser.parse_args()


def scan_gallery(gallery_dir: str, gallery_limit: int = 0) -> List[GalleryItem]:
    root = Path(gallery_dir)
    paths = sorted(root.rglob("*.png"))
    if gallery_limit > 0:
        paths = paths[:gallery_limit]
    items: List[GalleryItem] = []
    for idx, path in enumerate(paths):
        rel_path = path.relative_to(root).as_posix()
        parts = rel_path.split("/")
        image_class = parts[0] if len(parts) > 1 else path.stem
        items.append(
            GalleryItem(
                index=idx,
                filepath=str(path),
                rel_path=rel_path,
                filename=path.name,
                image_class=image_class,
            )
        )
    return items


def select_queries(items: Iterable[GalleryItem], max_queries_per_class: int, seed: int) -> List[int]:
    grouped: Dict[str, List[GalleryItem]] = defaultdict(list)
    for item in items:
        grouped[item.image_class].append(item)

    rng = random.Random(seed)
    selected_indices: List[int] = []
    for image_class, class_items in sorted(grouped.items()):
        if len(class_items) < 2:
            continue
        chosen = class_items
        if max_queries_per_class > 0 and len(class_items) > max_queries_per_class:
            chosen = rng.sample(class_items, max_queries_per_class)
        chosen = sorted(chosen, key=lambda item: item.filename)
        selected_indices.extend(item.index for item in chosen)
    return selected_indices


def safe_cosine_matrix(matrix: np.ndarray, vector: np.ndarray, matrix_norms: np.ndarray, vector_norm: float) -> np.ndarray:
    denom = np.maximum(matrix_norms * max(vector_norm, 1e-8), 1e-8)
    return np.clip((matrix @ vector) / denom, -1.0, 1.0)


def recall_at_k(ranked_indices: np.ndarray, query_idx: int, classes: List[str], query_class: str, k: int) -> float:
    top_indices = [idx for idx in ranked_indices if idx != query_idx][:k]
    return 1.0 if any(classes[idx] == query_class for idx in top_indices) else 0.0


def average_precision_at_k(
    ranked_indices: np.ndarray,
    query_idx: int,
    classes: List[str],
    query_class: str,
    top_k: int,
    relevant_total: int,
) -> float:
    if relevant_total <= 0:
        return 0.0
    hit_count = 0
    precision_sum = 0.0
    rank_position = 0
    for idx in ranked_indices:
        if idx == query_idx:
            continue
        rank_position += 1
        if rank_position > top_k:
            break
        if classes[idx] == query_class:
            hit_count += 1
            precision_sum += hit_count / rank_position
    denom = min(relevant_total, top_k)
    return precision_sum / denom if denom > 0 else 0.0


def ndcg_at_k(
    ranked_indices: np.ndarray,
    query_idx: int,
    classes: List[str],
    query_class: str,
    k: int,
    relevant_total: int,
) -> float:
    if relevant_total <= 0 or k <= 0:
        return 0.0
    dcg = 0.0
    rank_position = 0
    for idx in ranked_indices:
        if idx == query_idx:
            continue
        rank_position += 1
        if rank_position > k:
            break
        if classes[idx] == query_class:
            dcg += 1.0 / math.log2(rank_position + 1)
    ideal_hits = min(relevant_total, k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def tier_recall(
    ranked_indices: np.ndarray,
    query_idx: int,
    classes: List[str],
    query_class: str,
    relevant_total: int,
    multiplier: int,
) -> float:
    if relevant_total <= 0:
        return 0.0
    k = relevant_total * multiplier
    top_indices = [idx for idx in ranked_indices if idx != query_idx][:k]
    relevant_hits = sum(1 for idx in top_indices if classes[idx] == query_class)
    return relevant_hits / relevant_total


def anmrr(
    ranked_indices: np.ndarray,
    query_idx: int,
    classes: List[str],
    query_class: str,
    relevant_total: int,
    max_relevant: int,
) -> float:
    if relevant_total <= 0 or max_relevant <= 0:
        return 0.0
    k = min(4 * relevant_total, 2 * max_relevant)
    relevant_ranks = []
    rank_position = 0
    for idx in ranked_indices:
        if idx == query_idx:
            continue
        rank_position += 1
        if rank_position > k:
            break
        if classes[idx] == query_class:
            relevant_ranks.append(rank_position)
            if len(relevant_ranks) >= relevant_total:
                break
    while len(relevant_ranks) < relevant_total:
        relevant_ranks.append(k + 1)
    avr = sum(relevant_ranks) / relevant_total
    mrr = avr - 0.5 * (1 + relevant_total)
    denom = k + 0.5 - 0.5 * relevant_total
    return mrr / denom if denom > 0 else 0.0


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def to_markdown_table(rows: List[Dict], ordered_fields: List[str]) -> str:
    header = "| " + " | ".join(ordered_fields) + " |"
    separator = "| " + " | ".join(["---"] * len(ordered_fields)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(field, "")) for field in ordered_fields) + " |")
    return "\n".join([header, separator] + body)


def write_markdown_report(path: Path, metadata: Dict, summary_rows: List[Dict], per_class_rows: List[Dict]) -> None:
    lines = [
        "# 离线视觉检索评测结果",
        "",
        f"- 时间: `{metadata['timestamp']}`",
        f"- 数据集图片数: `{metadata['gallery_count']}`",
        f"- 查询数: `{metadata['query_count']}`",
        f"- 设备: `{metadata['device']}`",
        f"- 模式: `{', '.join(metadata['mode_names'])}`",
        "",
        "## 消融结果",
        "",
        to_markdown_table(
            summary_rows,
            ["label", "query_count", "recall_at_1", "recall_at_5", "recall_at_10", "map_at_k", "ndcg_at_k", "first_tier", "second_tier", "anmrr"],
        ),
        "",
        "## Full Model 分类结果",
        "",
        to_markdown_table(
            per_class_rows,
            ["class", "queries", "recall_at_1", "recall_at_5", "recall_at_10", "map_at_k", "ndcg_at_k", "first_tier", "second_tier", "anmrr"],
        ) if per_class_rows else "_No per-class rows_",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def mode_feature_key(mode_config: Dict) -> str:
    return f"clean_{int(mode_config['use_cleaning'])}_mask_{int(mode_config['use_masked_pooling'])}"


def descriptor_to_arrays(descriptor: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    return (
        np.asarray(descriptor.get("grid", []), dtype=np.float32),
        np.asarray(descriptor.get("horizontal_profile", []), dtype=np.float32),
        np.asarray(descriptor.get("vertical_profile", []), dtype=np.float32),
        float(descriptor.get("foreground_ratio", 0.0)),
    )


def build_cache_from_qdrant(cache_path: Path, gallery_items: List[GalleryItem]) -> Dict:
    init_vector_db()
    vector_db = get_vector_db()
    qdrant_db = vector_db.qdrant_db
    points_by_rel_path: Dict[str, Dict] = {}
    offset = None
    try:
        while True:
            points, offset = qdrant_db.client.scroll(
                collection_name=qdrant_db.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            for point in points:
                payload = point.payload or {}
                rel_path = payload.get("rel_path")
                if rel_path:
                    points_by_rel_path[rel_path] = {
                        "vector": np.asarray(point.vector, dtype=np.float32),
                        "payload": payload,
                    }
            if offset is None:
                break
    finally:
        vector_db.close()

    missing = [item.rel_path for item in gallery_items if item.rel_path not in points_by_rel_path]
    if missing:
        raise RuntimeError(f"Qdrant cache export missing {len(missing)} gallery items.")

    embeddings = []
    grid = []
    horizontal = []
    vertical = []
    ratio = []
    for item in gallery_items:
        record = points_by_rel_path[item.rel_path]
        payload = record["payload"]
        embeddings.append(record["vector"])
        grid.append(np.asarray(payload.get("structure_grid", []), dtype=np.float32))
        horizontal.append(np.asarray(payload.get("horizontal_profile", []), dtype=np.float32))
        vertical.append(np.asarray(payload.get("vertical_profile", []), dtype=np.float32))
        ratio.append(float(payload.get("foreground_ratio", 0.0)))

    cache = {
        "gallery_count": len(gallery_items),
        "embeddings": np.vstack(embeddings),
        "grid": np.vstack(grid),
        "horizontal": np.vstack(horizontal),
        "vertical": np.vstack(vertical),
        "ratio": np.asarray(ratio, dtype=np.float32),
        "embedding_norms": np.linalg.norm(np.vstack(embeddings), axis=1).astype(np.float32),
        "grid_norms": np.linalg.norm(np.vstack(grid), axis=1).astype(np.float32),
        "horizontal_norms": np.linalg.norm(np.vstack(horizontal), axis=1).astype(np.float32),
        "vertical_norms": np.linalg.norm(np.vstack(vertical), axis=1).astype(np.float32),
        "filenames": [item.filename for item in gallery_items],
        "filepaths": [item.filepath for item in gallery_items],
        "rel_paths": [item.rel_path for item in gallery_items],
        "classes": [item.image_class for item in gallery_items],
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as handle:
        pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return cache


def build_or_load_cache(
    *,
    cache_path: Path,
    gallery_items: List[GalleryItem],
    batch_size: int,
    feature_service,
    mode_config: Dict,
    rebuild_cache: bool,
) -> Dict:
    if cache_path.exists() and not rebuild_cache:
        with open(cache_path, "rb") as handle:
            cached = pickle.load(handle)
        if cached.get("gallery_count") == len(gallery_items):
            return cached

    if mode_config["use_cleaning"] and mode_config["use_masked_pooling"] and not rebuild_cache:
        print(f"Exporting cache from Qdrant for {cache_path.stem} ...", flush=True)
        return build_cache_from_qdrant(cache_path, gallery_items)

    embeddings_batches: List[np.ndarray] = []
    grid_batches: List[np.ndarray] = []
    horizontal_batches: List[np.ndarray] = []
    vertical_batches: List[np.ndarray] = []
    ratio_batches: List[np.ndarray] = []
    total_batches = max(1, math.ceil(len(gallery_items) / batch_size))

    for batch_index, start in enumerate(range(0, len(gallery_items), batch_size), start=1):
        batch_items = gallery_items[start : start + batch_size]
        images = []
        try:
            for item in batch_items:
                with Image.open(item.filepath) as image:
                    images.append(image.convert("RGB"))
            bundles = feature_service.extract_feature_bundles_batch(
                images,
                use_cleaning=mode_config["use_cleaning"],
                use_masked_pooling=mode_config["use_masked_pooling"],
                use_ocr_text=False,
            )
        finally:
            for image in images:
                image.close()

        batch_embeddings = []
        batch_grid = []
        batch_horizontal = []
        batch_vertical = []
        batch_ratio = []
        for bundle in bundles:
            grid, horizontal, vertical, ratio = descriptor_to_arrays(bundle["structure_descriptor"])
            batch_embeddings.append(np.asarray(bundle["embedding"], dtype=np.float32))
            batch_grid.append(grid)
            batch_horizontal.append(horizontal)
            batch_vertical.append(vertical)
            batch_ratio.append(ratio)

        embeddings_batches.append(np.vstack(batch_embeddings))
        grid_batches.append(np.vstack(batch_grid))
        horizontal_batches.append(np.vstack(batch_horizontal))
        vertical_batches.append(np.vstack(batch_vertical))
        ratio_batches.append(np.asarray(batch_ratio, dtype=np.float32))
        if batch_index == 1 or batch_index % 25 == 0 or batch_index == total_batches:
            print(
                f"  cache {cache_path.stem}: batch {batch_index}/{total_batches} ({min(start + batch_size, len(gallery_items))}/{len(gallery_items)})",
                flush=True,
            )

    embeddings = np.vstack(embeddings_batches)
    grid = np.vstack(grid_batches)
    horizontal = np.vstack(horizontal_batches)
    vertical = np.vstack(vertical_batches)
    ratio = np.concatenate(ratio_batches)

    cache = {
        "gallery_count": len(gallery_items),
        "embeddings": embeddings,
        "grid": grid,
        "horizontal": horizontal,
        "vertical": vertical,
        "ratio": ratio,
        "embedding_norms": np.linalg.norm(embeddings, axis=1).astype(np.float32),
        "grid_norms": np.linalg.norm(grid, axis=1).astype(np.float32),
        "horizontal_norms": np.linalg.norm(horizontal, axis=1).astype(np.float32),
        "vertical_norms": np.linalg.norm(vertical, axis=1).astype(np.float32),
        "filenames": [item.filename for item in gallery_items],
        "filepaths": [item.filepath for item in gallery_items],
        "rel_paths": [item.rel_path for item in gallery_items],
        "classes": [item.image_class for item in gallery_items],
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as handle:
        pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return cache


def compute_structure_similarity_batch(cache: Dict, query_idx: int) -> np.ndarray:
    query_grid = cache["grid"][query_idx]
    query_horizontal = cache["horizontal"][query_idx]
    query_vertical = cache["vertical"][query_idx]
    query_ratio = float(cache["ratio"][query_idx])

    grid_similarity = safe_cosine_matrix(cache["grid"], query_grid, cache["grid_norms"], float(np.linalg.norm(query_grid)))
    horizontal_similarity = safe_cosine_matrix(
        cache["horizontal"],
        query_horizontal,
        cache["horizontal_norms"],
        float(np.linalg.norm(query_horizontal)),
    )
    vertical_similarity = safe_cosine_matrix(
        cache["vertical"],
        query_vertical,
        cache["vertical_norms"],
        float(np.linalg.norm(query_vertical)),
    )
    density_similarity = np.maximum(0.0, 1.0 - np.abs(cache["ratio"] - query_ratio))
    score = 0.55 * grid_similarity + 0.15 * horizontal_similarity + 0.15 * vertical_similarity + 0.15 * density_similarity
    return np.clip(score, 0.0, 1.0)


def build_per_class_table(per_query_rows: List[Dict]) -> List[Dict]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in per_query_rows:
        grouped[row["query_class"]].append(row)
    table = []
    for image_class, rows in sorted(grouped.items()):
        count = len(rows)
        table.append(
            {
                "class": image_class,
                "queries": count,
                "recall_at_1": round(sum(item["recall_at_1"] for item in rows) / count, 4),
                "recall_at_5": round(sum(item["recall_at_5"] for item in rows) / count, 4),
                "recall_at_10": round(sum(item["recall_at_10"] for item in rows) / count, 4),
                "map_at_k": round(sum(item["ap_at_k"] for item in rows) / count, 4),
                "ndcg_at_k": round(sum(item["ndcg_at_k"] for item in rows) / count, 4),
                "first_tier": round(sum(item["first_tier"] for item in rows) / count, 4),
                "second_tier": round(sum(item["second_tier"] for item in rows) / count, 4),
                "anmrr": round(sum(item["anmrr"] for item in rows) / count, 4),
            }
        )
    return table


def evaluate_mode(
    *,
    mode_name: str,
    mode_config: Dict,
    cache: Dict,
    query_indices: List[int],
    top_k: int,
    metric_depth: int,
    class_counts: Dict[str, int],
    structure_weight: float,
) -> Dict:
    per_query_rows = []
    recall_hits = {1: 0.0, 5: 0.0, 10: 0.0}
    ap_scores = []
    ndcg_scores = []
    ft_scores = []
    st_scores = []
    anmrr_scores = []
    classes = cache["classes"]
    max_relevant = max((count - 1) for count in class_counts.values()) if class_counts else 0
    started_at = time.time()

    for query_idx in query_indices:
        query_class = classes[query_idx]
        relevant_total = max(0, class_counts.get(query_class, 0) - 1)

        semantic_cosine = cache["embeddings"] @ cache["embeddings"][query_idx]
        semantic_similarity = np.clip((semantic_cosine + 1.0) / 2.0, 0.0, 1.0)
        structure_similarity = (
            compute_structure_similarity_batch(cache, query_idx)
            if mode_config["enable_structure_rerank"]
            else np.zeros_like(semantic_similarity)
        )
        semantic_weight = 1.0 - structure_weight if mode_config["enable_structure_rerank"] else 1.0
        final_similarity = semantic_weight * semantic_similarity + structure_weight * structure_similarity
        final_similarity[query_idx] = -1.0
        ranked_indices = np.argsort(-final_similarity)[: metric_depth + 1]

        query_ap = average_precision_at_k(ranked_indices, query_idx, classes, query_class, top_k, relevant_total)
        query_ndcg = ndcg_at_k(ranked_indices, query_idx, classes, query_class, top_k, relevant_total)
        query_ft = tier_recall(ranked_indices, query_idx, classes, query_class, relevant_total, 1)
        query_st = tier_recall(ranked_indices, query_idx, classes, query_class, relevant_total, 2)
        query_anmrr = anmrr(ranked_indices, query_idx, classes, query_class, relevant_total, max_relevant)
        r1 = recall_at_k(ranked_indices, query_idx, classes, query_class, 1)
        r5 = recall_at_k(ranked_indices, query_idx, classes, query_class, 5)
        r10 = recall_at_k(ranked_indices, query_idx, classes, query_class, 10)

        top_idx = next(idx for idx in ranked_indices if idx != query_idx)
        per_query_rows.append(
            {
                "mode": mode_name,
                "query_class": query_class,
                "query_filename": cache["filenames"][query_idx],
                "query_filepath": cache["filepaths"][query_idx],
                "top1_class": classes[top_idx],
                "top1_filename": cache["filenames"][top_idx],
                "top1_filepath": cache["filepaths"][top_idx],
                "top1_similarity": round(float(final_similarity[top_idx]), 6),
                "top1_structure_similarity": round(float(structure_similarity[top_idx]), 6),
                "top1_is_relevant": int(classes[top_idx] == query_class),
                "recall_at_1": r1,
                "recall_at_5": r5,
                "recall_at_10": r10,
                "ap_at_k": round(query_ap, 6),
                "ndcg_at_k": round(query_ndcg, 6),
                "first_tier": round(query_ft, 6),
                "second_tier": round(query_st, 6),
                "anmrr": round(query_anmrr, 6),
            }
        )

        recall_hits[1] += r1
        recall_hits[5] += r5
        recall_hits[10] += r10
        ap_scores.append(query_ap)
        ndcg_scores.append(query_ndcg)
        ft_scores.append(query_ft)
        st_scores.append(query_st)
        anmrr_scores.append(query_anmrr)

    query_count = len(query_indices)
    summary = {
        "mode": mode_name,
        "label": mode_config["label"],
        "query_count": query_count,
        "recall_at_1": round(recall_hits[1] / query_count, 4),
        "recall_at_5": round(recall_hits[5] / query_count, 4),
        "recall_at_10": round(recall_hits[10] / query_count, 4),
        "map_at_k": round(sum(ap_scores) / query_count, 4),
        "ndcg_at_k": round(sum(ndcg_scores) / query_count, 4),
        "first_tier": round(sum(ft_scores) / query_count, 4),
        "second_tier": round(sum(st_scores) / query_count, 4),
        "anmrr": round(sum(anmrr_scores) / query_count, 4),
        "elapsed_seconds": round(time.time() - started_at, 2),
    }
    return {"summary": summary, "per_query_rows": per_query_rows}


def main() -> None:
    args = parse_args()
    os.environ["MODEL_DEVICE"] = args.device
    os.environ["INNOVATION_ENABLE_OCR_TEXT_FUSION"] = "false"
    init_config()
    init_feature_service()

    config = get_config()
    feature_service = get_feature_service()
    gallery_items = scan_gallery(config.gallery.cad_drawing_dir, gallery_limit=max(0, args.gallery_limit))
    if not gallery_items:
        raise RuntimeError("No gallery items found.")

    class_counts: Dict[str, int] = defaultdict(int)
    for item in gallery_items:
        class_counts[item.image_class] += 1

    query_indices = select_queries(gallery_items, max_queries_per_class=max(0, args.max_queries_per_class), seed=args.seed)
    if not query_indices:
        raise RuntimeError("No valid queries selected.")

    max_relevant = max((count - 1) for count in class_counts.values()) if class_counts else 0
    metric_depth = args.metric_depth if args.metric_depth > 0 else max(args.top_k, min(1000, 2 * max_relevant))

    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    selected_modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    for mode_name in selected_modes:
        if mode_name not in VISUAL_MODES:
            raise ValueError(f"Unknown visual mode: {mode_name}")

    feature_caches: Dict[str, Dict] = {}
    for mode_name in selected_modes:
        mode_config = VISUAL_MODES[mode_name]
        feature_key = mode_feature_key(mode_config)
        if feature_key in feature_caches:
            continue
        cache_path = cache_dir / f"{feature_key}.pkl"
        print(f"Preparing cache for {feature_key} ...", flush=True)
        feature_caches[feature_key] = build_or_load_cache(
            cache_path=cache_path,
            gallery_items=gallery_items,
            batch_size=args.batch_size,
            feature_service=feature_service,
            mode_config=mode_config,
            rebuild_cache=args.rebuild_cache,
        )

    summary_rows = []
    per_query_rows = []
    full_model_per_class_rows = []
    experiment_results = {}

    print(
        f"Loaded {len(gallery_items)} gallery images, selected {len(query_indices)} queries, metric depth={metric_depth}, device={feature_service.get_model_info().get('device')}.",
        flush=True,
    )

    for mode_name in selected_modes:
        mode_config = VISUAL_MODES[mode_name]
        feature_key = mode_feature_key(mode_config)
        result = evaluate_mode(
            mode_name=mode_name,
            mode_config=mode_config,
            cache=feature_caches[feature_key],
            query_indices=query_indices,
            top_k=args.top_k,
            metric_depth=metric_depth,
            class_counts=class_counts,
            structure_weight=config.innovation.structure_rerank_weight,
        )
        experiment_results[mode_name] = result
        summary_rows.append(result["summary"])
        per_query_rows.extend(result["per_query_rows"])
        if mode_name == "full_model":
            full_model_per_class_rows = build_per_class_table(result["per_query_rows"])

    metadata = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "gallery_count": len(gallery_items),
        "query_count": len(query_indices),
        "device": feature_service.get_model_info().get("device"),
        "mode_names": selected_modes,
        "top_k": args.top_k,
        "metric_depth": metric_depth,
        "max_queries_per_class": args.max_queries_per_class,
    }

    (output_dir / "retrieval_metrics_visual_offline.json").write_text(
        json.dumps(
            {
                "metadata": metadata,
                "summary_rows": summary_rows,
                "full_model_per_class_rows": full_model_per_class_rows,
                "experiment_results": experiment_results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_csv(
        output_dir / "ablation_table_visual_offline.csv",
        summary_rows,
        [
            "mode",
            "label",
            "query_count",
            "recall_at_1",
            "recall_at_5",
            "recall_at_10",
            "map_at_k",
            "ndcg_at_k",
            "first_tier",
            "second_tier",
            "anmrr",
            "elapsed_seconds",
        ],
    )
    write_csv(
        output_dir / "per_query_metrics_visual_offline.csv",
        per_query_rows,
        [
            "mode",
            "query_class",
            "query_filename",
            "query_filepath",
            "top1_class",
            "top1_filename",
            "top1_filepath",
            "top1_similarity",
            "top1_structure_similarity",
            "top1_is_relevant",
            "recall_at_1",
            "recall_at_5",
            "recall_at_10",
            "ap_at_k",
            "ndcg_at_k",
            "first_tier",
            "second_tier",
            "anmrr",
        ],
    )
    if full_model_per_class_rows:
        write_csv(
            output_dir / "full_model_per_class_visual_offline.csv",
            full_model_per_class_rows,
            [
                "class",
                "queries",
                "recall_at_1",
                "recall_at_5",
                "recall_at_10",
                "map_at_k",
                "ndcg_at_k",
                "first_tier",
                "second_tier",
                "anmrr",
            ],
        )
    write_markdown_report(output_dir / "visual_offline_report.md", metadata, summary_rows, full_model_per_class_rows)

    print("\nOffline visual ablation summary:", flush=True)
    for row in summary_rows:
        print(
            f"{row['label']}: R@1={row['recall_at_1']}, R@5={row['recall_at_5']}, "
            f"R@10={row['recall_at_10']}, mAP@{args.top_k}={row['map_at_k']}, "
            f"nDCG@{args.top_k}={row['ndcg_at_k']}, FT={row['first_tier']}, "
            f"ST={row['second_tier']}, ANMRR={row['anmrr']}",
            flush=True,
        )
    print(f"\nSaved outputs to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
