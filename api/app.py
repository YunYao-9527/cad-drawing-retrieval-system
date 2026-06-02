"""
FastAPI application for CAD drawing retrieval.
"""
from __future__ import annotations

import os
import time
import urllib.parse
from io import BytesIO

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

try:
    from starlette.templating import Jinja2Templates
except ImportError:  # pragma: no cover
    Jinja2Templates = None

from PIL import Image, ImageFile

from config.config_manager import get_config, init_config
from database.vector_db import get_vector_db
from health.health_check import get_health_checker
from monitoring.logger import get_logger, setup_logger
from monitoring.metrics import get_metrics_collector
from services.feature_service import get_feature_service
from services.retrieval_service import get_retrieval_service


def create_app() -> FastAPI:
    init_config()
    config = get_config()

    logger = setup_logger(
        name="api",
        level=config.logging.level,
        log_format=config.logging.format,
        log_file=config.logging.file,
        max_bytes=config.logging.max_bytes,
        backup_count=config.logging.backup_count,
        enable_console=config.logging.enable_console,
    )

    app = FastAPI(
        title=config.app.name,
        version=config.app.version,
        description="CAD 图纸智能检索系统 API",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        started_at = time.time()
        client_host = request.client.host if request.client else "unknown"
        logger.info(f"{request.method} {request.url.path} from {client_host}")
        try:
            response = await call_next(request)
            duration = time.time() - started_at
            get_metrics_collector().record_http_request(
                request.method,
                request.url.path,
                duration,
                response.status_code,
            )
            return response
        except Exception as exc:
            duration = time.time() - started_at
            logger.error(f"Request failed after {duration:.3f}s: {exc}", exc_info=True)
            raise

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "内部服务错误",
                "detail": str(exc) if config.app.debug else "请查看日志",
            },
        )

    templates_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")
    static_dir = os.path.join(templates_dir, "static")
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
    templates = Jinja2Templates(directory=templates_dir) if (Jinja2Templates and os.path.exists(templates_dir)) else None

    def generate_thumbnail(image_path: str, size=(400, 300)):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        with Image.open(image_path) as image:
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.thumbnail(size, Image.Resampling.LANCZOS)
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=85, optimize=True)
            buffer.seek(0)
            return StreamingResponse(buffer, media_type="image/jpeg")

    def parse_optional_bool(value):
        if value is None:
            return None
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
        raise HTTPException(status_code=400, detail=f"Invalid boolean value: {value}")

    @app.get("/")
    async def index(request: Request):
        if templates:
            return templates.TemplateResponse("index.html", {"request": request})
        return {"message": "Web 界面未配置"}

    @app.get("/health")
    async def health_check(detailed: bool = False):
        return get_health_checker().check_health(detailed=detailed)

    @app.get("/metrics")
    async def metrics():
        return Response(content=get_health_checker().get_metrics(), media_type="text/plain")

    @app.get("/ready")
    async def readiness():
        return get_health_checker().get_readiness()

    @app.get("/live")
    async def liveness():
        return get_health_checker().get_liveness()

    @app.get("/image/{filename:path}")
    async def serve_image(filename: str, thumb: bool = False):
        decoded_filename = urllib.parse.unquote(filename, encoding="utf-8").strip().replace("\\", "/")
        full_path = os.path.join(config.gallery.cad_drawing_dir, decoded_filename.replace("/", os.sep))

        if os.path.exists(full_path) and os.path.isfile(full_path):
            return generate_thumbnail(full_path) if thumb or os.path.getsize(full_path) > 2 * 1024 * 1024 else FileResponse(full_path)

        vector_db = get_vector_db()
        for image in vector_db.get_all_images():
            if image["id"] == decoded_filename:
                actual_path = image.get("filepath", "")
                if actual_path and os.path.exists(actual_path):
                    return (
                        generate_thumbnail(actual_path)
                        if thumb or os.path.getsize(actual_path) > 2 * 1024 * 1024
                        else FileResponse(actual_path)
                    )

        raise HTTPException(status_code=404, detail="图片未找到")

    @app.post("/search")
    async def search(
        image: UploadFile = File(...),
        top_k: int = Form(10),
        use_cleaning: str | None = Form(None),
        use_masked_pooling: str | None = Form(None),
        enable_structure_rerank: str | None = Form(None),
        use_ocr_text: str | None = Form(None),
        enable_text_fusion: str | None = Form(None),
        text_fusion_strategy: str | None = Form(None),
    ):
        if not image.filename or not image.filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            return JSONResponse(status_code=400, content={"status": "error", "message": "不支持的图片格式"})

        raw = await image.read()
        if not raw:
            return JSONResponse(status_code=400, content={"status": "error", "message": "图片内容为空"})

        try:
            input_image = Image.open(BytesIO(raw)).convert("RGB")
            parsed_use_cleaning = parse_optional_bool(use_cleaning)
            parsed_use_masked_pooling = parse_optional_bool(use_masked_pooling)
            parsed_enable_structure_rerank = parse_optional_bool(enable_structure_rerank)
            parsed_use_ocr_text = parse_optional_bool(use_ocr_text)
            parsed_enable_text_fusion = parse_optional_bool(enable_text_fusion)

            feature_bundle = get_feature_service().extract_feature_bundle(
                input_image,
                use_cleaning=parsed_use_cleaning,
                use_masked_pooling=parsed_use_masked_pooling,
                use_ocr_text=parsed_use_ocr_text,
            )
            results = get_retrieval_service().search(
                query_embedding=feature_bundle["embedding"],
                query_descriptor=feature_bundle["structure_descriptor"],
                query_text_descriptor=feature_bundle["text_descriptor"],
                query_text_embedding=feature_bundle.get("text_embedding"),
                query_filename=image.filename,
                top_k=top_k,
                enable_structure_rerank=parsed_enable_structure_rerank,
                enable_text_fusion=parsed_enable_text_fusion,
                text_fusion_strategy=text_fusion_strategy,
            )
            return {
                "status": "success",
                "query_image": image.filename,
                "experiment_flags": {
                    "use_cleaning": feature_bundle["experiment_flags"]["use_cleaning"],
                    "use_masked_pooling": feature_bundle["experiment_flags"]["use_masked_pooling"],
                    "enable_structure_rerank": (
                        config.innovation.enable_structure_rerank
                        if parsed_enable_structure_rerank is None
                        else parsed_enable_structure_rerank
                    ),
                    "use_ocr_text": feature_bundle["experiment_flags"]["use_ocr_text"],
                    "enable_text_fusion": (
                        config.innovation.enable_ocr_text_fusion
                        if parsed_enable_text_fusion is None
                        else parsed_enable_text_fusion
                    ),
                    "text_fusion_strategy": text_fusion_strategy or config.innovation.ocr_fusion_strategy,
                },
                "query_analysis": {
                    "mask_coverage": feature_bundle["cleaning_meta"].get("mask_coverage", 1.0),
                    "removed_regions": feature_bundle["cleaning_meta"].get("removed_regions", 0.0),
                    "ocr_text": feature_bundle["text_descriptor"].get("raw_text", ""),
                    "ocr_source": feature_bundle["text_descriptor"].get("source", "disabled"),
                    "title_block_entries": feature_bundle["text_descriptor"].get("title_block_entries", []),
                    "title_block_fields": feature_bundle["text_descriptor"].get("title_block_fields", {}),
                },
                "results": results,
            }
        except Exception as exc:
            logger.error(f"Search error: {exc}", exc_info=True)
            return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})

    @app.get("/api")
    async def api_info():
        return {
            "service": "CAD 图纸检索系统",
            "status": "running",
            "database_info": get_vector_db().get_stats(),
        }

    @app.get("/stats")
    async def stats():
        vector_db = get_vector_db()
        feature_service = get_feature_service()
        stats_data = vector_db.get_stats()
        images = vector_db.get_all_images()
        current_classes = {item["class"] for item in images if item.get("class")}
        model_info = feature_service.get_model_info()
        stats_data.update(
            {
                "device": model_info.get("device"),
                "feature_dim": config.model.embedding_dim,
                "current_categories": len(current_classes),
                "total_categories_in_model": len(model_info.get("class_names", []) or []),
            }
        )
        return stats_data

    @app.get("/api/images")
    async def get_images(page: int = 1, per_page: int = 20):
        images = get_vector_db().get_all_images()
        total = len(images)
        start = (page - 1) * per_page
        end = start + per_page
        return {
            "status": "success",
            "images": images[start:end],
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page,
        }

    @app.post("/api/images")
    async def add_image(image: UploadFile = File(...), image_class: str = Form(None)):
        if not image.filename:
            raise HTTPException(status_code=400, detail="文件名不能为空")

        image_class = image_class or os.path.splitext(image.filename)[0]
        class_dir = os.path.join(config.gallery.cad_drawing_dir, image_class)
        os.makedirs(class_dir, exist_ok=True)

        target_path = os.path.join(class_dir, image.filename)
        base_name, ext = os.path.splitext(image.filename)
        counter = 1
        while os.path.exists(target_path):
            target_path = os.path.join(class_dir, f"{base_name}_{counter}{ext}")
            counter += 1

        content = await image.read()
        with open(target_path, "wb") as handle:
            handle.write(content)

        if get_vector_db().add_image(target_path, image_class):
            return {"status": "success", "message": "图片添加成功"}

        try:
            os.remove(target_path)
        except OSError:
            pass
        raise HTTPException(status_code=500, detail="图片入库失败")

    @app.delete("/api/images/{image_id:path}")
    async def delete_image(image_id: str):
        if get_vector_db().delete_image(image_id):
            return {"status": "success", "message": "图片删除成功"}
        raise HTTPException(status_code=404, detail="图片不存在或删除失败")

    @app.get("/api/classes")
    async def get_classes():
        classes = sorted({image["class"] for image in get_vector_db().get_all_images() if image.get("class")})
        return {"status": "success", "classes": classes}

    @app.get("/api/categories")
    async def get_categories(page: int = 1, per_page: int = 20):
        all_images = get_vector_db().get_all_images()
        grouped = {}
        for image in all_images:
            category = image.get("class") or "未分类"
            grouped.setdefault(category, []).append(image)

        category_names = sorted(grouped.keys())
        total = len(category_names)
        start = (page - 1) * per_page
        end = start + per_page
        categories = [
            {
                "name": category,
                "count": len(grouped[category]),
                "preview_images": grouped[category][:4],
            }
            for category in category_names[start:end]
        ]
        return {
            "status": "success",
            "categories": categories,
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page,
        }

    @app.post("/api/initialize")
    async def initialize_database():
        count = get_vector_db().initialize_database()
        return {"status": "success", "message": f"数据库初始化完成，共处理 {count} 张图纸"}

    @app.post("/api/rebuild")
    async def rebuild_database():
        from qdrant_client.http import models

        vector_db = get_vector_db()
        vector_db.qdrant_db.client.delete(
            collection_name=vector_db.qdrant_db.collection_name,
            points_selector=models.FilterSelector(filter=models.Filter()),
        )
        count = vector_db.initialize_database()
        return {"status": "success", "message": f"数据库重建完成，共处理 {count} 张图纸"}

    @app.post("/api/cleanup")
    async def cleanup_database():
        deleted = get_vector_db().cleanup_duplicates()
        return {"status": "success", "message": f"清理完成，删除了 {deleted} 条重复记录"}

    @app.get("/api/find_duplicates")
    async def find_duplicates():
        duplicates = get_vector_db().find_duplicates()
        return {
            "status": "success",
            "duplicates": duplicates,
            "total_duplicate_groups": len(duplicates),
        }

    @app.get("/api/categories/{category_name}/images")
    async def get_category_images(category_name: str, page: int = 1, per_page: int = 20):
        images = [img for img in get_vector_db().get_all_images() if img.get("class") == category_name]
        total = len(images)
        start = (page - 1) * per_page
        end = start + per_page
        return {
            "status": "success",
            "images": images[start:end],
            "category": category_name,
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page,
        }

    @app.get("/api/debug/images")
    async def debug_images():
        images = get_vector_db().get_all_images()
        return {
            "status": "success",
            "total": len(images),
            "images": [
                {
                    "id": image["id"],
                    "filename": image["filename"],
                    "class": image["class"],
                    "exists": image.get("exists", False),
                    "mask_coverage": image.get("mask_coverage", 1.0),
                }
                for image in images
            ],
        }

    logger.info("API application created")
    return app
