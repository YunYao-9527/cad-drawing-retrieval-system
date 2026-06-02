"""
Formal retrieval evaluation script.

This script evaluates the running retrieval service through its HTTP API and
exports Recall@K / mAP tables suitable for a thesis or paper draft.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests


DEFAULT_MODES = {
    "baseline": {
        "label": "Baseline",
        "description": "No cleaning, no masked pooling, no structure rerank",
        "use_cleaning": False,
        "use_masked_pooling": False,
        "enable_structure_rerank": False,
    },
    "cleaning_only": {
        "label": "YOLO Cleaning Only",
        "description": "Enable YOLO cleaning only",
        "use_cleaning": True,
        "use_masked_pooling": False,
        "enable_structure_rerank": False,
    },
    "masked_pooling": {
        "label": "YOLO Cleaning + Masked Visual",
        "description": "Enable YOLO cleaning and mask-guided visual pooling",
        "use_cleaning": True,
        "use_masked_pooling": True,
        "enable_structure_rerank": False,
    },
    "full_model": {
        "label": "YOLO Cleaning + Structure",
        "description": "Enable YOLO cleaning, mask-guided visual pooling and structure rerank",
        "use_cleaning": True,
        "use_masked_pooling": True,
        "enable_structure_rerank": True,
        "use_ocr_text": False,
        "enable_text_fusion": False,
    },
    "multimodal_text": {
        "label": "YOLO Cleaning + Structure + OCR",
        "description": "Enable YOLO cleaning, structure rerank and title-block OCR text fusion",
        "use_cleaning": True,
        "use_masked_pooling": True,
        "enable_structure_rerank": True,
        "use_ocr_text": True,
        "enable_text_fusion": True,
        "text_fusion_strategy": "legacy",
    },
    "multimodal_hybrid": {
        "label": "+ OCR Hybrid Rerank",
        "description": "Enable cleaning, masked pooling, structure rerank and CLIP-guided OCR rerank",
        "use_cleaning": True,
        "use_masked_pooling": True,
        "enable_structure_rerank": True,
        "use_ocr_text": True,
        "enable_text_fusion": True,
        "text_fusion_strategy": "hybrid_rerank",
    },
    "no_cleaning_structure": {
        "label": "No Cleaning + Structure",
        "description": "Disable cleaning and masked pooling, keep structure rerank",
        "use_cleaning": False,
        "use_masked_pooling": False,
        "enable_structure_rerank": True,
        "use_ocr_text": False,
        "enable_text_fusion": False,
    },
    "no_cleaning_multimodal": {
        "label": "No Cleaning + OCR",
        "description": "Disable cleaning, keep structure rerank and OCR text fusion",
        "use_cleaning": False,
        "use_masked_pooling": False,
        "enable_structure_rerank": True,
        "use_ocr_text": True,
        "enable_text_fusion": True,
        "text_fusion_strategy": "hybrid_rerank",
    },
    "attention_visual": {
        "label": "Attention Visual (No Cleaning)",
        "description": "Disable YOLO cleaning and keep mask-guided attention pooling only",
        "use_cleaning": False,
        "use_masked_pooling": True,
        "enable_structure_rerank": False,
        "use_ocr_text": False,
        "enable_text_fusion": False,
    },
    "attention_structure": {
        "label": "Attention + Structure (No Cleaning)",
        "description": "Disable YOLO cleaning and enable mask-guided attention pooling plus structure rerank",
        "use_cleaning": False,
        "use_masked_pooling": True,
        "enable_structure_rerank": True,
        "use_ocr_text": False,
        "enable_text_fusion": False,
    },
    "attention_multimodal": {
        "label": "Attention + Structure + OCR (No Cleaning)",
        "description": "Disable YOLO cleaning and enable mask-guided attention pooling, structure rerank and OCR fusion",
        "use_cleaning": False,
        "use_masked_pooling": True,
        "enable_structure_rerank": True,
        "use_ocr_text": True,
        "enable_text_fusion": True,
        "text_fusion_strategy": "hybrid_rerank",
    },
}


def filtered_results(results: List[Dict], query: "QueryItem") -> List[Dict]:
    return [item for item in results if item.get("filepath") != query.filepath]


@dataclass
class QueryItem:
    image_id: str
    filename: str
    filepath: str
    image_class: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CAD retrieval service and export experiment tables.")
    parser.add_argument("--base-url", default="http://127.0.0.1:5000", help="Running retrieval service base URL")
    parser.add_argument("--top-k", type=int, default=10, help="Maximum retrieval depth used for metrics")
    parser.add_argument(
        "--max-queries-per-class",
        type=int,
        default=20,
        help="Balanced evaluation cap per class. Use 0 for all images.",
    )
    parser.add_argument(
        "--modes",
        default="baseline,cleaning_only,masked_pooling,full_model,multimodal_text,attention_visual,attention_structure,attention_multimodal",
        help="Comma-separated mode names",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for balanced sampling")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "results"),
        help="Directory for JSON/CSV/Markdown outputs",
    )
    parser.add_argument(
        "--metric-depth",
        type=int,
        default=0,
        help="Depth used for FT/ST/ANMRR evaluation. Use 0 for automatic depth.",
    )
    return parser.parse_args()


def fetch_all_images(base_url: str) -> List[QueryItem]:
    response = requests.get(f"{base_url}/api/images", params={"page": 1, "per_page": 10000}, timeout=120)
    response.raise_for_status()
    payload = response.json()
    images = []
    for item in payload.get("images", []):
        image_class = item.get("class")
        filepath = item.get("filepath")
        if not image_class or not filepath:
            continue
        images.append(
            QueryItem(
                image_id=item.get("id", ""),
                filename=item.get("filename", ""),
                filepath=filepath,
                image_class=image_class,
            )
        )
    return images


def select_queries(images: Iterable[QueryItem], max_queries_per_class: int, seed: int) -> List[QueryItem]:
    grouped: Dict[str, List[QueryItem]] = defaultdict(list)
    for item in images:
        grouped[item.image_class].append(item)

    rng = random.Random(seed)
    selected: List[QueryItem] = []
    for image_class, items in sorted(grouped.items()):
        if len(items) < 2:
            continue
        if max_queries_per_class and len(items) > max_queries_per_class:
            chosen = rng.sample(items, max_queries_per_class)
        else:
            chosen = list(items)
        chosen.sort(key=lambda item: item.filename)
        selected.extend(chosen)
    return selected


def average_precision_at_k(results: List[Dict], query: QueryItem, top_k: int, class_counts: Dict[str, int]) -> float:
    relevant_total = max(0, class_counts.get(query.image_class, 0) - 1)
    if relevant_total == 0:
        return 0.0

    filtered = filtered_results(results, query)[:top_k]
    hit_count = 0
    precision_sum = 0.0
    for rank, item in enumerate(filtered, start=1):
        is_relevant = item.get("class") == query.image_class
        if is_relevant:
            hit_count += 1
            precision_sum += hit_count / rank

    denom = min(relevant_total, top_k)
    if denom == 0:
        return 0.0
    return precision_sum / denom


def recall_at_k(results: List[Dict], query: QueryItem, k: int) -> float:
    filtered = filtered_results(results, query)[:k]
    return 1.0 if any(item.get("class") == query.image_class for item in filtered) else 0.0


def ndcg_at_k(results: List[Dict], query: QueryItem, k: int, class_counts: Dict[str, int]) -> float:
    relevant_total = max(0, class_counts.get(query.image_class, 0) - 1)
    if relevant_total == 0 or k <= 0:
        return 0.0

    filtered = filtered_results(results, query)[:k]
    dcg = 0.0
    for rank, item in enumerate(filtered, start=1):
        relevance = 1.0 if item.get("class") == query.image_class else 0.0
        if relevance > 0:
            dcg += relevance / math.log2(rank + 1)

    ideal_hits = min(relevant_total, k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def tier_recall(results: List[Dict], query: QueryItem, class_counts: Dict[str, int], multiplier: int) -> float:
    relevant_total = max(0, class_counts.get(query.image_class, 0) - 1)
    if relevant_total == 0:
        return 0.0
    k = relevant_total * multiplier
    filtered = filtered_results(results, query)[:k]
    relevant_hits = sum(1 for item in filtered if item.get("class") == query.image_class)
    return relevant_hits / relevant_total


def anmrr(results: List[Dict], query: QueryItem, class_counts: Dict[str, int], max_relevant: int) -> float:
    ng = max(0, class_counts.get(query.image_class, 0) - 1)
    if ng == 0 or max_relevant <= 0:
        return 0.0

    k = min(4 * ng, 2 * max_relevant)
    filtered = filtered_results(results, query)[:k]
    relevant_ranks = []
    for rank, item in enumerate(filtered, start=1):
        if item.get("class") == query.image_class:
            relevant_ranks.append(rank)
            if len(relevant_ranks) >= ng:
                break

    while len(relevant_ranks) < ng:
        relevant_ranks.append(k + 1)

    avr = sum(relevant_ranks) / ng
    mrr = avr - 0.5 * (1 + ng)
    denom = k + 0.5 - 0.5 * ng
    if denom <= 0:
        return 0.0
    return mrr / denom


def evaluate_mode(
    base_url: str,
    mode_name: str,
    mode_config: Dict,
    queries: List[QueryItem],
    top_k: int,
    class_counts: Dict[str, int],
    metric_depth: int,
) -> Dict:
    session = requests.Session()
    per_query_rows = []
    recall_hits = {1: 0.0, 5: 0.0, 10: 0.0}
    ap_scores = []
    ndcg_scores = []
    ft_scores = []
    st_scores = []
    anmrr_scores = []
    started_at = time.time()
    max_relevant = max((count - 1) for count in class_counts.values()) if class_counts else 0

    for index, query in enumerate(queries, start=1):
        with open(query.filepath, "rb") as handle:
            response = session.post(
                f"{base_url}/search",
                files={"image": (query.filename, handle, "image/png")},
                data={
                    "top_k": metric_depth,
                    "use_cleaning": str(mode_config["use_cleaning"]).lower(),
                    "use_masked_pooling": str(mode_config["use_masked_pooling"]).lower(),
                    "enable_structure_rerank": str(mode_config["enable_structure_rerank"]).lower(),
                    "use_ocr_text": str(mode_config.get("use_ocr_text", False)).lower(),
                    "enable_text_fusion": str(mode_config.get("enable_text_fusion", False)).lower(),
                    "text_fusion_strategy": mode_config.get("text_fusion_strategy", ""),
                },
                timeout=180,
            )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results", [])

        query_ap = average_precision_at_k(results, query, top_k=top_k, class_counts=class_counts)
        query_ndcg = ndcg_at_k(results, query, k=top_k, class_counts=class_counts)
        query_ft = tier_recall(results, query, class_counts=class_counts, multiplier=1)
        query_st = tier_recall(results, query, class_counts=class_counts, multiplier=2)
        query_anmrr = anmrr(results, query, class_counts=class_counts, max_relevant=max_relevant)
        ap_scores.append(query_ap)
        ndcg_scores.append(query_ndcg)
        ft_scores.append(query_ft)
        st_scores.append(query_st)
        anmrr_scores.append(query_anmrr)
        for recall_k in recall_hits:
            recall_hits[recall_k] += recall_at_k(results, query, recall_k)

        filtered = filtered_results(results, query)
        top1 = filtered[0] if filtered else {}
        per_query_rows.append(
            {
                "mode": mode_name,
                "query_class": query.image_class,
                "query_filename": query.filename,
                "query_filepath": query.filepath,
                "top1_class": top1.get("class"),
                "top1_filename": top1.get("filename"),
                "top1_filepath": top1.get("filepath"),
                "top1_similarity": round(float(top1.get("similarity", 0.0)), 6) if top1 else 0.0,
                "top1_structure_similarity": round(float(top1.get("structure_similarity", 0.0)), 6) if top1 else 0.0,
                "top1_is_relevant": 1 if top1.get("class") == query.image_class and top1 else 0,
                "recall_at_1": recall_at_k(results, query, 1),
                "recall_at_5": recall_at_k(results, query, 5),
                "recall_at_10": recall_at_k(results, query, 10),
                "ap_at_k": round(query_ap, 6),
                "ndcg_at_k": round(query_ndcg, 6),
                "first_tier": round(query_ft, 6),
                "second_tier": round(query_st, 6),
                "anmrr": round(query_anmrr, 6),
            }
        )

        if index % 25 == 0 or index == len(queries):
            print(
                f"[{mode_name}] processed {index}/{len(queries)} queries",
                flush=True,
            )

    query_count = len(queries)
    summary = {
        "mode": mode_name,
        "label": mode_config["label"],
        "description": mode_config["description"],
        "query_count": query_count,
        "recall_at_1": round(recall_hits[1] / query_count, 4) if query_count else 0.0,
        "recall_at_5": round(recall_hits[5] / query_count, 4) if query_count else 0.0,
        "recall_at_10": round(recall_hits[10] / query_count, 4) if query_count else 0.0,
        "map_at_k": round(sum(ap_scores) / query_count, 4) if query_count else 0.0,
        "ndcg_at_k": round(sum(ndcg_scores) / query_count, 4) if query_count else 0.0,
        "first_tier": round(sum(ft_scores) / query_count, 4) if query_count else 0.0,
        "second_tier": round(sum(st_scores) / query_count, 4) if query_count else 0.0,
        "anmrr": round(sum(anmrr_scores) / query_count, 4) if query_count else 0.0,
        "elapsed_seconds": round(time.time() - started_at, 2),
        "flags": {
            "use_cleaning": mode_config["use_cleaning"],
            "use_masked_pooling": mode_config["use_masked_pooling"],
            "enable_structure_rerank": mode_config["enable_structure_rerank"],
            "use_ocr_text": mode_config.get("use_ocr_text", False),
            "enable_text_fusion": mode_config.get("enable_text_fusion", False),
            "text_fusion_strategy": mode_config.get("text_fusion_strategy", ""),
        },
    }
    return {"summary": summary, "per_query_rows": per_query_rows}


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


def write_markdown_report(
    path: Path,
    *,
    metadata: Dict,
    summary_rows: List[Dict],
    per_class_rows: List[Dict],
) -> None:
    summary_table = to_markdown_table(
        summary_rows,
        [
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
        ],
    )
    per_class_table = to_markdown_table(
        per_class_rows,
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
    lines = [
        "# 检索评测结果",
        "",
        f"- 评测时间: `{metadata['timestamp']}`",
        f"- 服务地址: `{metadata['base_url']}`",
        f"- Top-K: `{metadata['top_k']}`",
        f"- Metric Depth: `{metadata['metric_depth']}`",
        f"- 查询样本数: `{metadata['query_count']}`",
        f"- 每类最大查询数: `{metadata['max_queries_per_class']}`",
        "",
        "## 消融实验表",
        "",
        summary_table,
        "",
        "## Full Model 分类别结果",
        "",
        per_class_table,
        "",
        "## 备注",
        "",
        "- 当前评测采用固定图库索引。",
        f"- 本次实验模式: `{', '.join(metadata['mode_names'])}`。",
        "- 推荐将结果按“YOLO 清洗路线”与“无清洗注意力路线”两条主线进行讨论。",
        "- `mAP` 统计口径为 `mAP@K`，其中 `K = Top-K`。",
        "- `nDCG` 采用二值相关性定义，数值越大越好。",
        "- `FT/ST` 分别表示 First Tier 与 Second Tier 召回率。",
        "- `ANMRR` 按 MPEG-7 图像检索口径计算，数值越小越好。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_modes = []
    for mode_name in [item.strip() for item in args.modes.split(",") if item.strip()]:
        if mode_name not in DEFAULT_MODES:
            raise ValueError(f"Unknown mode: {mode_name}")
        selected_modes.append(mode_name)

    all_images = fetch_all_images(base_url)
    class_counts: Dict[str, int] = defaultdict(int)
    for item in all_images:
        class_counts[item.image_class] += 1

    queries = select_queries(
        all_images,
        max_queries_per_class=max(0, args.max_queries_per_class),
        seed=args.seed,
    )
    if not queries:
        raise RuntimeError("No valid evaluation queries were found.")

    max_relevant = max((count - 1) for count in class_counts.values()) if class_counts else 0
    metric_depth = args.metric_depth if args.metric_depth > 0 else max(args.top_k, min(1000, 2 * max_relevant))

    print(f"Loaded {len(all_images)} indexed images and selected {len(queries)} evaluation queries.", flush=True)
    print(f"Using metric depth = {metric_depth}", flush=True)

    experiment_results = {}
    summary_rows = []
    per_query_rows = []
    full_model_per_class_rows = []

    for mode_name in selected_modes:
        result = evaluate_mode(
            base_url=base_url,
            mode_name=mode_name,
            mode_config=DEFAULT_MODES[mode_name],
            queries=queries,
            top_k=args.top_k,
            class_counts=class_counts,
            metric_depth=metric_depth,
        )
        experiment_results[mode_name] = result
        summary_rows.append(result["summary"])
        per_query_rows.extend(result["per_query_rows"])
        if mode_name in {"full_model", "multimodal_text"}:
            full_model_per_class_rows = build_per_class_table(result["per_query_rows"])

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metadata = {
        "timestamp": timestamp,
        "base_url": base_url,
        "top_k": args.top_k,
        "metric_depth": metric_depth,
        "query_count": len(queries),
        "max_queries_per_class": args.max_queries_per_class,
        "mode_names": selected_modes,
    }

    json_path = output_dir / "retrieval_metrics.json"
    csv_summary_path = output_dir / "ablation_table.csv"
    csv_per_query_path = output_dir / "per_query_metrics.csv"
    csv_per_class_path = output_dir / "full_model_per_class.csv"
    md_path = output_dir / "ablation_report.md"

    json_path.write_text(
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
        csv_summary_path,
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
        csv_per_query_path,
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
            csv_per_class_path,
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
    write_markdown_report(
        md_path,
        metadata=metadata,
        summary_rows=summary_rows,
        per_class_rows=full_model_per_class_rows,
    )

    print("\nAblation summary:", flush=True)
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
