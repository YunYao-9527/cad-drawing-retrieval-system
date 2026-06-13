"""
CAD 图纸检索系统 - 轻量 Demo 模式
跳过 PyTorch/CLIP/YOLO，只用 Pillow 提供 Web UI 展示和模拟检索结果。
适用于 Render Free Tier (512MB RAM) 等资源受限环境。
"""
from __future__ import annotations

import os
import random
import time
import urllib.parse
from io import BytesIO

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageFile

try:
    from starlette.templating import Jinja2Templates
except ImportError:
    Jinja2Templates = None

# ---------------------------------------------------------------------------
# In-memory image catalog (no Qdrant / PyTorch needed)
# ---------------------------------------------------------------------------

IMAGE_CATALOG: list[dict] = []
GALLERY_DIR = os.environ.get("GALLERY_DIR", "./demo_gallery")


def scan_gallery():
    """Scan gallery directory for images and populate IMAGE_CATALOG."""
    global IMAGE_CATALOG
    IMAGE_CATALOG = []
    if not os.path.isdir(GALLERY_DIR):
        os.makedirs(GALLERY_DIR, exist_ok=True)
        return

    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    for root, _dirs, files in os.walk(GALLERY_DIR):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in supported:
                continue
            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, GALLERY_DIR).replace("\\", "/")
            category = rel_path.split("/")[0] if "/" in rel_path else "未分类"
            try:
                fsize = os.path.getsize(full_path)
            except OSError:
                fsize = 0
            IMAGE_CATALOG.append({
                "id": rel_path,
                "filename": fname,
                "class": category,
                "filepath": full_path,
                "filesize": fsize,
                "exists": True,
            })


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_demo_app() -> FastAPI:
    app = FastAPI(
        title="CAD图纸检索系统 (Demo)",
        version="2.1.0-demo",
        description="CAD 图纸智能检索系统 - 轻量演示模式（无 ML 推理）",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    templates_dir = os.path.join(os.path.dirname(__file__), "templates")
    static_dir = os.path.join(templates_dir, "static")
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
    templates = (
        Jinja2Templates(directory=templates_dir)
        if (Jinja2Templates and os.path.exists(templates_dir))
        else None
    )

    def _thumbnail(path: str, size=(400, 300)):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        with Image.open(path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.thumbnail(size, Image.Resampling.LANCZOS)
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=85, optimize=True)
            buf.seek(0)
            return StreamingResponse(buf, media_type="image/jpeg")

    # ---- Routes ----

    @app.get("/")
    async def index(request: Request):
        if templates:
            return templates.TemplateResponse("index.html", {"request": request})
        return {"message": "Web 界面未配置"}

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "mode": "demo",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_images": len(IMAGE_CATALOG),
        }

    @app.get("/metrics")
    async def metrics():
        return Response(content="# Demo mode - no metrics", media_type="text/plain")

    @app.get("/ready")
    async def readiness():
        return {"status": "ready"}

    @app.get("/live")
    async def liveness():
        return {"status": "alive"}

    @app.get("/image/{filename:path}")
    async def serve_image(filename: str, thumb: bool = False):
        decoded = urllib.parse.unquote(filename, encoding="utf-8").strip().replace("\\", "/")
        # Try direct path
        full_path = os.path.join(GALLERY_DIR, decoded.replace("/", os.sep))
        if os.path.isfile(full_path):
            return _thumbnail(full_path) if thumb else FileResponse(full_path)
        # Search catalog
        for item in IMAGE_CATALOG:
            if item["id"] == decoded and os.path.isfile(item["filepath"]):
                return _thumbnail(item["filepath"]) if thumb else FileResponse(item["filepath"])
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
        """Return random results from the catalog (demo mode - no ML inference)."""
        if not IMAGE_CATALOG:
            return {
                "status": "success",
                "query_image": image.filename,
                "experiment_flags": {
                    "use_cleaning": False,
                    "use_masked_pooling": False,
                    "enable_structure_rerank": False,
                    "use_ocr_text": False,
                    "enable_text_fusion": False,
                    "text_fusion_strategy": "demo",
                },
                "query_analysis": {
                    "mask_coverage": 1.0,
                    "removed_regions": 0,
                    "ocr_text": "",
                    "ocr_source": "demo",
                    "title_block_entries": [],
                    "title_block_fields": {},
                },
                "results": [],
            }

        k = min(top_k, len(IMAGE_CATALOG))
        sampled = random.sample(IMAGE_CATALOG, k)
        # Generate descending similarity scores
        base = random.uniform(0.85, 0.98)
        results = []
        for rank, item in enumerate(sampled, 1):
            sim = max(0.3, base - (rank - 1) * random.uniform(0.02, 0.06))
            results.append({
                "id": item["id"],
                "filename": item["filename"],
                "class": item["class"],
                "similarity": round(sim, 4),
                "distance": round(1.0 - sim, 4),
                "rank": rank,
            })

        return {
            "status": "success",
            "query_image": image.filename,
            "experiment_flags": {
                "use_cleaning": False,
                "use_masked_pooling": False,
                "enable_structure_rerank": False,
                "use_ocr_text": False,
                "enable_text_fusion": False,
                "text_fusion_strategy": "demo",
            },
            "query_analysis": {
                "mask_coverage": 1.0,
                "removed_regions": 0,
                "ocr_text": "",
                "ocr_source": "demo",
                "title_block_entries": [],
                "title_block_fields": {},
            },
            "results": results,
        }

    @app.get("/api")
    async def api_info():
        return {
            "service": "CAD 图纸检索系统 (Demo)",
            "status": "running",
            "mode": "demo",
            "total_images": len(IMAGE_CATALOG),
        }

    @app.get("/stats")
    async def stats():
        classes = {item["class"] for item in IMAGE_CATALOG}
        class_dist = {}
        for item in IMAGE_CATALOG:
            cls = item["class"]
            class_dist[cls] = class_dist.get(cls, 0) + 1
        return {
            "total_images": len(IMAGE_CATALOG),
            "device": "CPU (Demo Mode)",
            "database_path": "in-memory",
            "current_categories": len(classes),
            "total_categories_in_model": len(classes),
            "feature_dim": 512,
            "model_classes": len(classes),
            "class_distribution": class_dist,
        }

    @app.get("/api/images")
    async def get_images(page: int = 1, per_page: int = 20):
        total = len(IMAGE_CATALOG)
        start = (page - 1) * per_page
        end = start + per_page
        return {
            "status": "success",
            "images": IMAGE_CATALOG[start:end],
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
        class_dir = os.path.join(GALLERY_DIR, image_class)
        os.makedirs(class_dir, exist_ok=True)
        target_path = os.path.join(class_dir, image.filename)
        base, ext = os.path.splitext(image.filename)
        counter = 1
        while os.path.exists(target_path):
            target_path = os.path.join(class_dir, f"{base}_{counter}{ext}")
            counter += 1
        content = await image.read()
        with open(target_path, "wb") as f:
            f.write(content)
        rel_path = os.path.relpath(target_path, GALLERY_DIR).replace("\\", "/")
        IMAGE_CATALOG.append({
            "id": rel_path,
            "filename": image.filename,
            "class": image_class,
            "filepath": target_path,
            "filesize": len(content),
            "exists": True,
        })
        return {"status": "success", "message": "图片添加成功"}

    @app.delete("/api/images/{image_id:path}")
    async def delete_image(image_id: str):
        decoded = urllib.parse.unquote(image_id, encoding="utf-8")
        for i, item in enumerate(IMAGE_CATALOG):
            if item["id"] == decoded:
                try:
                    os.remove(item["filepath"])
                except OSError:
                    pass
                IMAGE_CATALOG.pop(i)
                return {"status": "success", "message": "图片删除成功"}
        raise HTTPException(status_code=404, detail="图片不存在")

    @app.get("/api/classes")
    async def get_classes():
        classes = sorted({item["class"] for item in IMAGE_CATALOG})
        return {"status": "success", "classes": classes}

    @app.get("/api/categories")
    async def get_categories(page: int = 1, per_page: int = 20):
        grouped: dict[str, list] = {}
        for item in IMAGE_CATALOG:
            grouped.setdefault(item["class"], []).append(item)
        names = sorted(grouped.keys())
        total = len(names)
        start = (page - 1) * per_page
        end = start + per_page
        categories = [
            {"name": n, "count": len(grouped[n]), "preview_images": grouped[n][:4]}
            for n in names[start:end]
        ]
        return {
            "status": "success",
            "categories": categories,
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page,
        }

    @app.get("/api/categories/{category_name}/images")
    async def get_category_images(category_name: str, page: int = 1, per_page: int = 20):
        images = [img for img in IMAGE_CATALOG if img["class"] == category_name]
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

    @app.post("/api/initialize")
    async def initialize_database():
        scan_gallery()
        return {"status": "success", "message": f"数据库初始化完成，共处理 {len(IMAGE_CATALOG)} 张图纸"}

    @app.post("/api/rebuild")
    async def rebuild_database():
        scan_gallery()
        return {"status": "success", "message": f"数据库重建完成，共处理 {len(IMAGE_CATALOG)} 张图纸"}

    @app.post("/api/cleanup")
    async def cleanup_database():
        return {"status": "success", "message": "Demo 模式下无需清理"}

    @app.get("/api/find_duplicates")
    async def find_duplicates():
        return {"status": "success", "duplicates": {}, "total_duplicate_groups": 0}

    @app.get("/api/debug/images")
    async def debug_images():
        return {
            "status": "success",
            "total": len(IMAGE_CATALOG),
            "images": [
                {"id": i["id"], "filename": i["filename"], "class": i["class"], "exists": i["exists"]}
                for i in IMAGE_CATALOG
            ],
        }

    return app


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("CAD 图纸检索系统 - 轻量 Demo 模式")
    print("跳过 PyTorch/CLIP/YOLO，仅提供 Web UI 展示")
    print("=" * 60)

    scan_gallery()
    print(f"扫描到 {len(IMAGE_CATALOG)} 张图片")

    app = create_demo_app()
    host = os.environ.get("APP_HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", os.environ.get("APP_PORT", "5000")))

    print(f"启动服务: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
