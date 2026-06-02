"""
Application entrypoint.
"""
from __future__ import annotations

import os

import uvicorn

from api.app import create_app
from config.config_manager import get_config, init_config
from database.vector_db import get_vector_db, init_vector_db
from monitoring.logger import setup_logger
from services.cleaning_service import init_cleaning_service
from services.feature_service import init_feature_service
from services.ocr_service import init_ocr_service


def main():
    init_config()
    config = get_config()

    logger = setup_logger(
        name="main",
        level=config.logging.level,
        log_format=config.logging.format,
        log_file=config.logging.file,
        max_bytes=config.logging.max_bytes,
        backup_count=config.logging.backup_count,
        enable_console=config.logging.enable_console,
    )

    logger.info("=" * 60)
    logger.info(f"Starting {config.app.name} v{config.app.version}")
    logger.info("=" * 60)

    if config.cleaning.enabled:
        default_model_path = config.cleaning.model_path or r"runs\detect\drawing_cleaner_v1\weights\best.pt"
        if default_model_path and os.path.exists(default_model_path):
            init_cleaning_service(default_model_path)
        else:
            logger.warning(f"Cleaning model not found, rule fallback will be used: {default_model_path}")

    if config.innovation.enable_ocr_text_fusion:
        init_ocr_service()

    logger.info("Initializing feature service...")
    init_feature_service()

    logger.info("Initializing vector database...")
    init_vector_db()
    vector_db = get_vector_db()

    db_count = vector_db.qdrant_db.get_count()
    if db_count == 0:
        logger.warning("Empty vector database detected, starting full index build.")
        processed = vector_db.initialize_database()
        logger.info(f"Initial indexing completed, processed {processed} drawings.")
    else:
        stats = vector_db.get_stats()
        logger.info(f"Vector database already exists with {stats.get('total_images', 0)} drawings.")

    app = create_app()

    logger.info(f"Gallery path: {config.gallery.cad_drawing_dir}")
    logger.info(f"API docs: http://{config.app.host}:{config.app.port}/docs")
    logger.info(
        "Innovation switches: masked_pooling=%s, structure_rerank=%s, ocr_text_fusion=%s, adapter_type=%s",
        config.innovation.enable_masked_pooling,
        config.innovation.enable_structure_rerank,
        config.innovation.enable_ocr_text_fusion,
        config.innovation.adapter_type,
    )

    uvicorn.run(
        app,
        host=config.app.host,
        port=config.app.port,
        log_level=config.logging.level.lower(),
        access_log=config.logging.enable_console,
    )


if __name__ == "__main__":
    main()
