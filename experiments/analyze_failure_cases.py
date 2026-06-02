"""
Generate representative false-positive case studies for thesis writing.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze representative retrieval failure cases.")
    parser.add_argument("--per-query-csv", required=True, help="Path to per_query_metrics.csv")
    parser.add_argument("--mode", required=True, help="Mode name to analyze")
    parser.add_argument("--case-count", type=int, default=6, help="Maximum number of failure cases")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "failure_cases"),
        help="Directory for markdown and generated visualizations",
    )
    parser.add_argument(
        "--generate-visuals",
        action="store_true",
        help="Generate pairwise similarity visualizations for selected cases",
    )
    return parser.parse_args()


def load_rows(csv_path: Path) -> List[Dict]:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def select_cases(rows: List[Dict], mode: str, case_count: int) -> List[Dict]:
    filtered = [row for row in rows if row.get("mode") == mode and row.get("top1_is_relevant") == "0"]
    filtered.sort(key=lambda row: float(row.get("top1_similarity", 0.0)), reverse=True)

    selected = []
    used_classes = {}
    for row in filtered:
        query_class = row.get("query_class", "")
        if used_classes.get(query_class, 0) >= 2:
            continue
        selected.append(row)
        used_classes[query_class] = used_classes.get(query_class, 0) + 1
        if len(selected) >= case_count:
            break
    return selected


def classify_reason(row: Dict) -> str:
    structure_similarity = float(row.get("top1_structure_similarity", 0.0) or 0.0)
    similarity = float(row.get("top1_similarity", 0.0) or 0.0)

    if structure_similarity >= 0.95:
        return "两张图的全局布局和主视图线框非常接近，结构重排仍将其判为高相似候选。"
    if structure_similarity >= 0.80:
        return "局部轮廓和构图模式存在较强重叠，但类别级语义差异没有被充分拉开。"
    if similarity >= 0.98:
        return "视觉主干特征对模板化边框、剖面线或标题栏布局较敏感，导致跨类高分误检。"
    return "图纸线条较稀疏或局部区域主导特征过强，使得检索结果偏向外观相近类别。"


def build_visualization(row: Dict, output_dir: Path) -> Path | None:
    query_path = row.get("query_filepath", "")
    target_path = row.get("top1_filepath", "")
    if not query_path or not target_path:
        return None

    script_path = Path(__file__).resolve().parent / "visualize_pair_similarity.py"
    project_root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--query",
            query_path,
            "--target",
            target_path,
            "--output-dir",
            str(output_dir),
            "--method",
            "grad_cam",
        ],
        check=True,
        cwd=project_root,
        env=env,
    )
    prefix = f"{Path(query_path).stem}__VS__{Path(target_path).stem}"
    png_path = output_dir / f"{prefix}.png"
    return png_path if png_path.exists() else None


def write_markdown(path: Path, mode: str, cases: List[Dict], visual_paths: Dict[str, Path | None]) -> None:
    lines = [
        f"# 误检案例分析（{mode}）",
        "",
        f"- 案例数量: `{len(cases)}`",
        "- 选取规则: 按错误 Top-1 的相似度从高到低排序，并控制类别分布。",
        "",
    ]

    for index, row in enumerate(cases, start=1):
        query_path = row.get("query_filepath", "")
        target_path = row.get("top1_filepath", "")
        lines.extend(
            [
                f"## 案例 {index}",
                "",
                f"- 查询类别: `{row.get('query_class', '')}`",
                f"- 误检类别: `{row.get('top1_class', '')}`",
                f"- 查询图: `{row.get('query_filename', '')}`",
                f"- 误检图: `{row.get('top1_filename', '')}`",
                f"- Top-1 相似度: `{row.get('top1_similarity', '')}`",
                f"- Top-1 结构相似度: `{row.get('top1_structure_similarity', '')}`",
                f"- 原因判断: {classify_reason(row)}",
                "",
                f"查询图：![query]({query_path})" if query_path else "查询图：路径缺失",
                "",
                f"误检图：![target]({target_path})" if target_path else "误检图：路径缺失",
                "",
            ]
        )
        visual_path = visual_paths.get(row.get("query_filepath", ""))
        if visual_path:
            lines.extend([f"成对相似性图：![pair]({visual_path})", ""])

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    csv_path = Path(args.per_query_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(csv_path)
    cases = select_cases(rows, mode=args.mode, case_count=args.case_count)
    if not cases:
        raise RuntimeError(f"No failure cases found for mode: {args.mode}")

    visual_paths: Dict[str, Path | None] = {}
    if args.generate_visuals:
        for row in cases:
            visual_paths[row.get("query_filepath", "")] = build_visualization(row, output_dir)
    else:
        visual_paths = {row.get("query_filepath", ""): None for row in cases}

    report_path = output_dir / f"failure_cases_{args.mode}.md"
    write_markdown(report_path, args.mode, cases, visual_paths)
    print(f"Saved failure case report to: {report_path}")


if __name__ == "__main__":
    main()
