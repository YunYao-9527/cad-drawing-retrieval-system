"""
Standalone gallery index builder.

Use this script when you want to build or rebuild the vector index without
starting the FastAPI service. It also supports disabling OCR-heavy branches so
visual-only experiments can finish much faster and keep CPU usage under better
control.
"""
from __future__ import annotations

import argparse
import os

from config.config_manager import get_config, init_config
from database.vector_db import get_vector_db, init_vector_db
from monitoring.logger import setup_logger
from services.cleaning_service import init_cleaning_service
from services.feature_service import get_feature_service, init_feature_service
from services.ocr_service import init_ocr_service


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build or rebuild the CAD drawing vector index.")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="cuda",
        help="Model device. Default is cuda to avoid CPU-only indexing when a GPU is available.",
    )
    parser.add_argument(
        "--disable-ocr",
        action="store_true",
        help="Disable OCR text fusion during indexing. Recommended for the first visual-only full build.",
    )
    parser.add_argument(
        "--disable-cleaning",
        action="store_true",
        help="Disable YOLO cleaning during indexing.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override vector upsert batch size.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete the current Qdrant collection and rebuild it from scratch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Initialize services and print the effective runtime configuration without building the index.",
    )
    return parser


def apply_runtime_overrides(args: argparse.Namespace) -> None:
    os.environ["MODEL_DEVICE"] = args.device
    if args.disable_ocr:
        os.environ["INNOVATION_ENABLE_OCR_TEXT_FUSION"] = "false"
    if args.disable_cleaning:
        os.environ["CLEANING_ENABLED"] = "false"


def main() -> None:
    args = build_parser().parse_args()
    apply_runtime_overrides(args)

    init_config()
    config = get_config()

    logger = setup_logger(
        name="build_index",
        level=config.logging.level,
        log_format=config.logging.format,
        log_file=config.logging.file,
        max_bytes=config.logging.max_bytes,
        backup_count=config.logging.backup_count,
        enable_console=config.logging.enable_console,
    )

    logger.info("=" * 60)
    logger.info("Starting standalone index build")
    logger.info("=" * 60)
    logger.info(f"Gallery path: {config.gallery.cad_drawing_dir}")
    logger.info(
        "Runtime switches: device=%s, cleaning=%s, masked_pooling=%s, structure_rerank=%s, ocr_text_fusion=%s",
        config.model.device,
        config.cleaning.enabled,
        config.innovation.enable_masked_pooling,
        config.innovation.enable_structure_rerank,
        config.innovation.enable_ocr_text_fusion,
    )

    if config.cleaning.enabled:
        default_model_path = config.cleaning.model_path or r"runs\detect\drawing_cleaner_v1\weights\best.pt"
        if default_model_path and os.path.exists(default_model_path):
            init_cleaning_service(default_model_path)
        else:
            logger.warning(f"Cleaning model not found, cleaning will be skipped: {default_model_path}")

    if config.innovation.enable_ocr_text_fusion:
        init_ocr_service()
    else:
        logger.info("OCR text fusion disabled for this indexing run.")

    init_feature_service()
    feature_service = get_feature_service()
    logger.info(f"Feature extractor device: {feature_service.get_model_info().get('device')}")

    init_vector_db()
    vector_db = get_vector_db()

    try:
        if args.dry_run:
            stats = vector_db.get_stats()
            logger.info(
                "Dry run completed: current_total_images=%s, database=%s",
                stats.get("total_images", 0),
                stats.get("database_path", "unknown"),
            )
            return

        if args.rebuild:
            logger.warning("Rebuild requested, deleting the current collection before indexing.")
            vector_db.reset_collection()

        processed = vector_db.initialize_database(batch_size=args.batch_size)
        stats = vector_db.get_stats()
        logger.info(
            "Index build finished: processed=%s, total_images=%s",
            processed,
            stats.get("total_images", 0),
        )
    finally:
        vector_db.close()


if __name__ == "__main__":
    main()
