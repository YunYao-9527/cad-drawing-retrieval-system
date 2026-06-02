"""
Offline multimodal rerank evaluation built on top of visual caches.

This script evaluates the two OCR-assisted routes:
1. YOLO Cleaning + Structure + OCR
2. Attention + Structure + OCR (No Cleaning)

It keeps the gallery/query visual features aligned with the visual offline
evaluation, then applies OCR/title-block reranking only to top candidates,
which matches the actual system design more closely than full-gallery OCR.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config_manager import get_config, init_config
from experiments.evaluate_visual_offline import (
    VISUAL_MODES,
    anmrr,
    average_precision_at_k,
    build_per_class_table,
    compute_structure_similarity_batch,
    ndcg_at_k,
    recall_at_k,
    scan_gallery,
    select_queries,
    tier_recall,
)
from services.feature_service import get_feature_service, init_feature_service
from services.ocr_service import (
    analyze_text_descriptors,
    build_filename_descriptor,
    compare_text_descriptors,
    compare_title_block_fields,
    get_ocr_service,
    init_ocr_service,
)


MULTIMODAL_MODES = {
    "multimodal_text": {
        "label": "YOLO Cleaning + Structure + OCR",
        "base_mode": "full_model",
        "use_cleaning": True,
        "text_fusion_strategy": "legacy",
    },
    "attention_multimodal": {
        "label": "Attention + Structure + OCR (No Cleaning)",
        "base_mode": "attention_structure",
        "use_cleaning": False,
        "text_fusion_strategy": "hybrid_rerank",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline OCR rerank evaluation on top of visual caches.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--metric-depth", type=int, default=0)
    parser.add_argument("--max-queries-per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modes", default="multimodal_text,attention_multimodal")
    parser.add_argument(
        "--cache-dir",
        default=str(Path(__file__).resolve().parent / "cache_visual_dataset_img3"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "results_multimodal_offline_dataset_img3"),
    )
    parser.add_argument("--gallery-limit", type=int, default=0)
    return parser.parse_args()


def load_visual_cache(cache_dir: Path, mode_name: str) -> Dict:
    mode = VISUAL_MODES[mode_name]
    cache_key = f"clean_{int(mode['use_cleaning'])}_mask_{int(mode['use_masked_pooling'])}"
    cache_path = cache_dir / f"{cache_key}.pkl"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing visual cache: {cache_path}")
    import pickle

    with open(cache_path, "rb") as handle:
        return pickle.load(handle)


def build_base_scores(cache: Dict, query_idx: int, structure_weight: float, enable_structure_rerank: bool) -> Tuple[np.ndarray, np.ndarray]:
    semantic_cosine = cache["embeddings"] @ cache["embeddings"][query_idx]
    semantic_similarity = np.clip((semantic_cosine + 1.0) / 2.0, 0.0, 1.0)
    structure_similarity = (
        compute_structure_similarity_batch(cache, query_idx) if enable_structure_rerank else np.zeros_like(semantic_similarity)
    )
    semantic_weight = 1.0 - structure_weight if enable_structure_rerank else 1.0
    final_similarity = semantic_weight * semantic_similarity + structure_weight * structure_similarity
    final_similarity[query_idx] = -1.0
    return final_similarity, structure_similarity


def empty_text_result() -> Dict:
    return {
        "descriptor": {
            "raw_text": "",
            "text_lines": [],
            "title_block_lines": [],
            "title_block_entries": [],
            "tokens": [],
            "number_tokens": [],
            "identifier_tokens": [],
            "chargrams": [],
            "has_text": False,
            "source": "disabled",
            "text_quality": 0.0,
            "title_block_fields": {
                "part_name": "",
                "drawing_no": "",
                "material": "",
                "scale": "",
                "quantity": "",
                "field_quality": 0.0,
                "kv_pair_count": 0,
                "detection_method": "rules",
                "field_sources": {},
            },
        },
        "filename_descriptor": build_filename_descriptor(""),
    }


def get_text_payload(
    cache: Dict,
    index: int,
    *,
    use_cleaning: bool,
    feature_service,
    ocr_service,
    text_cache: Dict[Tuple[int, bool], Dict],
) -> Dict:
    key = (index, use_cleaning)
    if key in text_cache:
        return text_cache[key]

    filepath = cache["filepaths"][index]
    filename = cache["filenames"][index]
    try:
        with Image.open(filepath) as image:
            rgb = image.convert("RGB")
            if use_cleaning:
                _, mask_image, _ = feature_service._apply_cleaning(rgb)
            else:
                mask_image = feature_service._fallback_keep_mask(rgb)
            descriptor = ocr_service.extract_text_descriptor(rgb, mask_image=mask_image)
    except Exception:
        result = empty_text_result()
        result["filename_descriptor"] = build_filename_descriptor(filename)
        text_cache[key] = result
        return result

    result = {
        "descriptor": descriptor,
        "filename_descriptor": build_filename_descriptor(filename),
    }
    text_cache[key] = result
    return result


def apply_ocr_rerank(
    *,
    base_scores: np.ndarray,
    structure_scores: np.ndarray,
    cache: Dict,
    query_idx: int,
    use_cleaning: bool,
    strategy: str,
    feature_service,
    ocr_service,
    text_cache: Dict[Tuple[int, bool], Dict],
    config,
) -> np.ndarray:
    reranked = base_scores.copy()
    ranked_indices = np.argsort(-base_scores)
    rerank_topn = min(len(ranked_indices), max(config.innovation.ocr_rerank_topn, 30))

    query_payload = get_text_payload(
        cache,
        query_idx,
        use_cleaning=use_cleaning,
        feature_service=feature_service,
        ocr_service=ocr_service,
        text_cache=text_cache,
    )
    query_text_descriptor = query_payload["descriptor"]
    query_filename_descriptor = query_payload["filename_descriptor"]
    if not query_text_descriptor.get("has_text"):
        return reranked

    query_title_fields = query_text_descriptor.get("title_block_fields")
    query_filename_fields = query_filename_descriptor.get("title_block_fields")

    for candidate_idx in ranked_indices[:rerank_topn]:
        if candidate_idx == query_idx:
            continue

        target_payload = get_text_payload(
            cache,
            candidate_idx,
            use_cleaning=use_cleaning,
            feature_service=feature_service,
            ocr_service=ocr_service,
            text_cache=text_cache,
        )
        target_text_descriptor = target_payload["descriptor"]
        target_filename_descriptor = target_payload["filename_descriptor"]

        text_similarity = compare_text_descriptors(query_text_descriptor, target_text_descriptor)
        text_analysis = analyze_text_descriptors(query_text_descriptor, target_text_descriptor)
        field_analysis = compare_title_block_fields(
            query_title_fields,
            target_text_descriptor.get("title_block_fields"),
        )
        filename_field_analysis = compare_title_block_fields(
            query_filename_fields,
            target_filename_descriptor.get("title_block_fields"),
        )
        field_similarity = max(
            float(field_analysis.get("field_score", 0.0)),
            0.85 * float(filename_field_analysis.get("field_score", 0.0)),
        )

        final_similarity = float(base_scores[candidate_idx])
        text_bonus = 0.0
        field_bonus = 0.0
        identifier_bonus = 0.0
        partial_identifier_bonus = 0.0
        filename_identifier_bonus = 0.0

        if strategy == "hybrid_rerank":
            filename_analysis = analyze_text_descriptors(query_filename_descriptor, target_filename_descriptor)
            filename_similarity = float(filename_analysis.get("fusion_score", 0.0))
            lexical_anchor = max(
                text_similarity,
                filename_similarity,
                float(text_analysis.get("identifier_similarity", 0.0)),
                float(filename_analysis.get("identifier_similarity", 0.0)),
            )
            combined_text_similarity = 0.75 * text_similarity + 0.25 * filename_similarity
            reliable_quality = max(
                float(text_analysis.get("quality", 0.0)),
                0.8 * float(filename_analysis.get("quality", 0.0)),
            )
            reliable_gain = 0.0
            if (
                lexical_anchor >= 0.16
                or text_analysis.get("exact_identifier_match", 0.0) > 0
                or filename_analysis.get("exact_identifier_match", 0.0) > 0
                or text_analysis.get("partial_identifier_match", 0.0) >= 0.94
                or filename_analysis.get("partial_identifier_match", 0.0) >= 0.94
            ):
                reliable_gain = combined_text_similarity * reliable_quality
            if reliable_gain > config.innovation.ocr_bonus_threshold:
                normalized_bonus = (reliable_gain - config.innovation.ocr_bonus_threshold) / max(
                    1e-6,
                    1.0 - config.innovation.ocr_bonus_threshold,
                )
                text_bonus = config.innovation.ocr_text_weight * normalized_bonus
            if text_analysis.get("exact_identifier_match", 0.0) > 0:
                identifier_bonus = config.innovation.ocr_identifier_bonus * float(text_analysis.get("quality", 0.0))
            elif text_analysis.get("partial_identifier_match", 0.0) >= 0.92:
                partial_identifier_bonus = config.innovation.ocr_partial_identifier_bonus * float(text_analysis.get("quality", 0.0))
            if filename_analysis.get("exact_identifier_match", 0.0) > 0:
                filename_identifier_bonus = 0.5 * config.innovation.ocr_identifier_bonus
            elif filename_analysis.get("partial_identifier_match", 0.0) >= 0.92:
                filename_identifier_bonus = 0.5 * config.innovation.ocr_partial_identifier_bonus
        else:
            if (
                text_similarity > config.innovation.ocr_bonus_threshold
                and float(text_analysis.get("quality", 0.0)) >= 0.35
                and (
                    float(text_analysis.get("identifier_similarity", 0.0)) >= 0.05
                    or float(text_analysis.get("exact_identifier_match", 0.0)) > 0
                    or float(text_analysis.get("line_similarity", 0.0)) >= 0.65
                )
            ):
                normalized_bonus = (text_similarity - config.innovation.ocr_bonus_threshold) / max(
                    1e-6,
                    1.0 - config.innovation.ocr_bonus_threshold,
                )
                text_bonus = config.innovation.ocr_text_weight * normalized_bonus

            if (
                field_similarity > 0.10
                and (
                    float(field_analysis.get("trusted_field_coverage", 0.0)) >= 0.20
                    or float(field_analysis.get("drawing_no_trusted", 0.0)) > 0
                    or float(field_analysis.get("part_name_trusted", 0.0)) > 0
                )
            ):
                quality_gate = max(
                    0.55,
                    float(field_analysis.get("field_quality", 0.0)),
                    0.85 * float(filename_field_analysis.get("field_quality", 0.0)),
                )
                field_bonus = 0.7 * config.innovation.ocr_text_weight * field_similarity * quality_gate
            if (
                field_analysis.get("exact_drawing_no_match", 0.0) > 0
                or filename_field_analysis.get("exact_drawing_no_match", 0.0) > 0
            ) and field_analysis.get("drawing_no_trusted", 0.0) > 0:
                field_bonus += 0.75 * config.innovation.ocr_identifier_bonus
            elif (
                field_analysis.get("drawing_no_similarity", 0.0) >= 0.92
                or filename_field_analysis.get("drawing_no_similarity", 0.0) >= 0.92
            ) and field_analysis.get("drawing_no_trusted", 0.0) > 0:
                field_bonus += 0.5 * config.innovation.ocr_partial_identifier_bonus
            if (
                field_analysis.get("part_name_similarity", 0.0) >= 0.97
                and field_analysis.get("part_name_trusted", 0.0) > 0
            ):
                field_bonus += 0.35 * config.innovation.ocr_identifier_bonus
            elif (
                field_analysis.get("part_name_similarity", 0.0) >= 0.90
                and field_analysis.get("part_name_trusted", 0.0) > 0
            ):
                field_bonus += 0.2 * config.innovation.ocr_partial_identifier_bonus
            if (
                field_analysis.get("material_similarity", 0.0) >= 0.98
                and field_analysis.get("scale_similarity", 0.0) >= 0.98
            ):
                field_bonus += 0.1 * config.innovation.ocr_identifier_bonus

        final_similarity += text_bonus + field_bonus + identifier_bonus + partial_identifier_bonus + filename_identifier_bonus
        reranked[candidate_idx] = max(0.0, min(1.0, final_similarity))

    reranked[query_idx] = -1.0
    return reranked


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    os.environ["MODEL_DEVICE"] = args.device
    os.environ["INNOVATION_ENABLE_OCR_TEXT_FUSION"] = "true"
    init_config()
    config = get_config()
    init_feature_service()
    init_ocr_service()
    feature_service = get_feature_service()
    ocr_service = get_ocr_service()

    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gallery_items = scan_gallery(config.gallery.cad_drawing_dir, gallery_limit=max(0, args.gallery_limit))
    class_counts: Dict[str, int] = {}
    for item in gallery_items:
        class_counts[item.image_class] = class_counts.get(item.image_class, 0) + 1
    query_indices = select_queries(gallery_items, max_queries_per_class=max(0, args.max_queries_per_class), seed=args.seed)
    max_relevant = max((count - 1) for count in class_counts.values()) if class_counts else 0
    metric_depth = args.metric_depth if args.metric_depth > 0 else max(args.top_k, min(1000, 2 * max_relevant))

    selected_modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    for mode_name in selected_modes:
        if mode_name not in MULTIMODAL_MODES:
            raise ValueError(f"Unknown multimodal mode: {mode_name}")

    summary_rows = []
    per_query_rows = []
    experiment_results = {}
    per_class_rows = {}
    text_cache: Dict[Tuple[int, bool], Dict] = {}

    for mode_name in selected_modes:
        mode = MULTIMODAL_MODES[mode_name]
        base_mode = VISUAL_MODES[mode["base_mode"]]
        cache = load_visual_cache(cache_dir, mode["base_mode"])
        recall_hits = {1: 0.0, 5: 0.0, 10: 0.0}
        ap_scores = []
        ndcg_scores = []
        ft_scores = []
        st_scores = []
        anmrr_scores = []
        mode_query_rows = []
        started_at = time.time()

        print(f"Evaluating {mode_name} ...", flush=True)
        for query_number, query_idx in enumerate(query_indices, start=1):
            if query_number == 1 or query_number % 50 == 0 or query_number == len(query_indices):
                print(f"  {mode_name}: query {query_number}/{len(query_indices)}", flush=True)

            base_scores, structure_scores = build_base_scores(
                cache,
                query_idx,
                structure_weight=config.innovation.structure_rerank_weight,
                enable_structure_rerank=base_mode["enable_structure_rerank"],
            )
            reranked_scores = apply_ocr_rerank(
                base_scores=base_scores,
                structure_scores=structure_scores,
                cache=cache,
                query_idx=query_idx,
                use_cleaning=mode["use_cleaning"],
                strategy=mode["text_fusion_strategy"],
                feature_service=feature_service,
                ocr_service=ocr_service,
                text_cache=text_cache,
                config=config,
            )
            ranked_indices = np.argsort(-reranked_scores)[: metric_depth + 1]
            query_class = cache["classes"][query_idx]
            relevant_total = max(0, class_counts.get(query_class, 0) - 1)

            query_ap = average_precision_at_k(ranked_indices, query_idx, cache["classes"], query_class, args.top_k, relevant_total)
            query_ndcg = ndcg_at_k(ranked_indices, query_idx, cache["classes"], query_class, args.top_k, relevant_total)
            query_ft = tier_recall(ranked_indices, query_idx, cache["classes"], query_class, relevant_total, 1)
            query_st = tier_recall(ranked_indices, query_idx, cache["classes"], query_class, relevant_total, 2)
            query_anmrr = anmrr(ranked_indices, query_idx, cache["classes"], query_class, relevant_total, max_relevant)
            r1 = recall_at_k(ranked_indices, query_idx, cache["classes"], query_class, 1)
            r5 = recall_at_k(ranked_indices, query_idx, cache["classes"], query_class, 5)
            r10 = recall_at_k(ranked_indices, query_idx, cache["classes"], query_class, 10)
            top_idx = next(idx for idx in ranked_indices if idx != query_idx)

            row = {
                "mode": mode_name,
                "query_class": query_class,
                "query_filename": cache["filenames"][query_idx],
                "query_filepath": cache["filepaths"][query_idx],
                "top1_class": cache["classes"][top_idx],
                "top1_filename": cache["filenames"][top_idx],
                "top1_filepath": cache["filepaths"][top_idx],
                "top1_similarity": round(float(reranked_scores[top_idx]), 6),
                "top1_structure_similarity": round(float(structure_scores[top_idx]), 6),
                "top1_is_relevant": int(cache["classes"][top_idx] == query_class),
                "recall_at_1": r1,
                "recall_at_5": r5,
                "recall_at_10": r10,
                "ap_at_k": round(query_ap, 6),
                "ndcg_at_k": round(query_ndcg, 6),
                "first_tier": round(query_ft, 6),
                "second_tier": round(query_st, 6),
                "anmrr": round(query_anmrr, 6),
            }
            mode_query_rows.append(row)
            per_query_rows.append(row)
            recall_hits[1] += r1
            recall_hits[5] += r5
            recall_hits[10] += r10
            ap_scores.append(query_ap)
            ndcg_scores.append(query_ndcg)
            ft_scores.append(query_ft)
            st_scores.append(query_st)
            anmrr_scores.append(query_anmrr)

        count = len(query_indices)
        summary = {
            "mode": mode_name,
            "label": mode["label"],
            "query_count": count,
            "recall_at_1": round(recall_hits[1] / count, 4),
            "recall_at_5": round(recall_hits[5] / count, 4),
            "recall_at_10": round(recall_hits[10] / count, 4),
            "map_at_k": round(sum(ap_scores) / count, 4),
            "ndcg_at_k": round(sum(ndcg_scores) / count, 4),
            "first_tier": round(sum(ft_scores) / count, 4),
            "second_tier": round(sum(st_scores) / count, 4),
            "anmrr": round(sum(anmrr_scores) / count, 4),
            "elapsed_seconds": round(time.time() - started_at, 2),
        }
        summary_rows.append(summary)
        per_class_rows[mode_name] = build_per_class_table(mode_query_rows)
        experiment_results[mode_name] = {"summary": summary, "per_query_rows": mode_query_rows}

    metadata = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": feature_service.get_model_info().get("device"),
        "gallery_count": len(gallery_items),
        "query_count": len(query_indices),
        "mode_names": selected_modes,
        "top_k": args.top_k,
        "metric_depth": metric_depth,
    }
    (output_dir / "retrieval_metrics_multimodal_offline.json").write_text(
        json.dumps(
            {
                "metadata": metadata,
                "summary_rows": summary_rows,
                "per_class_rows": per_class_rows,
                "experiment_results": experiment_results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_csv(
        output_dir / "ablation_table_multimodal_offline.csv",
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
        output_dir / "per_query_metrics_multimodal_offline.csv",
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

    print("\nOffline multimodal summary:", flush=True)
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
