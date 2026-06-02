from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageChops, ImageOps


DEFAULT_DARK_THRESHOLD = 40
DEFAULT_MIN_DARK_POINTS = 6
DEFAULT_GAMMA = 1.6


@dataclass
class ImageProcessResult:
    path: str
    converted: bool
    background: str
    width: int
    height: int


def iter_png_files(root: Path) -> Iterable[Path]:
    return root.rglob("*.png")


def build_sample_points(width: int, height: int) -> list[tuple[int, int]]:
    max_x = max(width - 1, 0)
    max_y = max(height - 1, 0)
    mid_x = width // 2
    mid_y = height // 2
    quarter_x = width // 4
    three_quarter_x = (width * 3) // 4
    quarter_y = height // 4
    three_quarter_y = (height * 3) // 4

    points = [
        (0, 0),
        (mid_x, 0),
        (max_x, 0),
        (0, mid_y),
        (max_x, mid_y),
        (0, max_y),
        (mid_x, max_y),
        (max_x, max_y),
        (quarter_x, 0),
        (three_quarter_x, 0),
        (quarter_x, max_y),
        (three_quarter_x, max_y),
        (0, quarter_y),
        (0, three_quarter_y),
        (max_x, quarter_y),
        (max_x, three_quarter_y),
    ]
    return list(dict.fromkeys(points))


def is_black_background(image: Image.Image, dark_threshold: int, min_dark_points: int) -> bool:
    rgb = image.convert("RGB")
    dark_points = 0

    for x, y in build_sample_points(*rgb.size):
        r, g, b = rgb.getpixel((x, y))
        if (r + g + b) / 3 <= dark_threshold:
            dark_points += 1

    return dark_points >= min_dark_points


def apply_gamma(image: Image.Image, gamma: float) -> Image.Image:
    lut = [
        max(0, min(255, int(round(255 * ((value / 255) ** gamma)))))
        for value in range(256)
    ]
    return image.point(lut)


def convert_black_background(image: Image.Image, gamma: float) -> Image.Image:
    rgb = image.convert("RGB")
    red, green, blue = rgb.split()
    max_channel = ImageChops.lighter(ImageChops.lighter(red, green), blue)
    normalized = ImageOps.invert(max_channel)
    normalized = apply_gamma(normalized, gamma=gamma)
    return normalized.convert("RGB")


def normalize_dataset(
    root: Path,
    dry_run: bool,
    dark_threshold: int,
    min_dark_points: int,
    gamma: float,
    progress_every: int,
) -> tuple[list[ImageProcessResult], int]:
    results: list[ImageProcessResult] = []
    converted_count = 0

    for index, path in enumerate(iter_png_files(root), start=1):
        with Image.open(path) as image:
            width, height = image.size
            black_background = is_black_background(
                image=image,
                dark_threshold=dark_threshold,
                min_dark_points=min_dark_points,
            )

            if black_background:
                converted_count += 1
                if not dry_run:
                    normalized = convert_black_background(image, gamma=gamma)
                    normalized.save(path)
                results.append(
                    ImageProcessResult(
                        path=str(path),
                        converted=True,
                        background="black",
                        width=width,
                        height=height,
                    )
                )
            else:
                results.append(
                    ImageProcessResult(
                        path=str(path),
                        converted=False,
                        background="white_or_mixed",
                        width=width,
                        height=height,
                    )
                )

        if progress_every > 0 and index % progress_every == 0:
            print(
                f"[progress] scanned={index} converted={converted_count}",
                flush=True,
            )

    return results, converted_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="统一工程图 PNG 背景为白底。")
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="PNG 数据集根目录",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="输出 JSON 报告路径",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只统计，不实际修改文件",
    )
    parser.add_argument(
        "--dark-threshold",
        type=int,
        default=DEFAULT_DARK_THRESHOLD,
        help="判定黑底的亮度阈值，默认 40",
    )
    parser.add_argument(
        "--min-dark-points",
        type=int,
        default=DEFAULT_MIN_DARK_POINTS,
        help="边界采样点中至少多少个为暗色才判定为黑底，默认 6",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=DEFAULT_GAMMA,
        help="线稿增强系数，默认 1.6；越大线条越深",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="每处理多少张图输出一次进度，默认 500",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.exists():
        parser.error(f"数据集目录不存在: {root}")

    results, converted_count = normalize_dataset(
        root=root,
        dry_run=args.dry_run,
        dark_threshold=args.dark_threshold,
        min_dark_points=args.min_dark_points,
        gamma=args.gamma,
        progress_every=args.progress_every,
    )

    summary = {
        "root": str(root),
        "total_png": len(results),
        "converted_black_background_png": converted_count,
        "dry_run": args.dry_run,
        "dark_threshold": args.dark_threshold,
        "min_dark_points": args.min_dark_points,
        "gamma": args.gamma,
        "progress_every": args.progress_every,
    }

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": summary,
            "files": [asdict(item) for item in results],
        }
        args.report.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
