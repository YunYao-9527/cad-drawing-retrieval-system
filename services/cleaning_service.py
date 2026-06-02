"""
Drawing cleaning service.

The service now returns both a cleaned image and a binary keep-mask so the
feature extractor can perform mask-aware pooling/attention.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from config.config_manager import get_config
from monitoring.logger import get_logger

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - optional dependency
    YOLO = None

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


class CleaningService:
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger("cleaning_service")
        self.model = None
        self._is_initialized = False

    def initialize(self, model_path: Optional[str] = None):
        path = model_path or self.config.cleaning.model_path
        if not self.config.cleaning.enabled:
            self.logger.info("图纸清洗已在配置中关闭")
            self._is_initialized = False
            return
        if not path:
            self.logger.warning("未配置清洗模型路径，图纸清洗将退化为规则掩码")
            self._is_initialized = False
            return
        if YOLO is None:
            self.logger.warning("未安装 ultralytics，图纸清洗将退化为规则掩码")
            self._is_initialized = False
            return

        try:
            self.logger.info(f"正在加载图纸清洗模型: {path}")
            self.model = YOLO(path)
            self._is_initialized = True
            self.logger.info("图纸清洗模型加载完成")
        except Exception as exc:  # pragma: no cover - runtime dependency
            self.logger.error(f"图纸清洗模型加载失败: {exc}", exc_info=True)
            self._is_initialized = False

    def _default_mask(self, image: Image.Image) -> Image.Image:
        keep = np.full((image.height, image.width), 255, dtype=np.uint8)
        return Image.fromarray(keep, mode="L")

    def _predict_boxes(self, image_bgr: np.ndarray) -> np.ndarray:
        if not self._is_initialized or self.model is None:
            return np.zeros((0, 4), dtype=np.int32)

        if torch is not None and torch.cuda.is_available():
            yolo_device = 0
        else:
            yolo_device = "cpu"

        result = self.model.predict(
            image_bgr,
            conf=self.config.cleaning.confidence,
            iou=self.config.cleaning.iou,
            device=yolo_device,
            verbose=False,
        )
        boxes = []
        for prediction in result:
            if getattr(prediction, "boxes", None) is None:
                continue
            for box in prediction.boxes.xyxy.cpu().numpy():
                boxes.append(box.astype(np.int32))
        return np.asarray(boxes, dtype=np.int32) if boxes else np.zeros((0, 4), dtype=np.int32)

    def clean_image_with_mask(
        self,
        image: Image.Image,
    ) -> Tuple[Image.Image, Image.Image, Dict[str, float]]:
        """Return cleaned image, keep-mask and cleaning metadata."""
        if image is None:
            return image, None, {"removed_regions": 0.0, "mask_coverage": 0.0}

        try:
            image_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
            height, width = image_bgr.shape[:2]
            removal_mask = np.zeros((height, width), dtype=np.uint8)
            boxes = self._predict_boxes(image_bgr)
            padding = max(0, int(self.config.cleaning.padding))

            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(width, x2 + padding)
                y2 = min(height, y2 + padding)
                cv2.rectangle(removal_mask, (x1, y1), (x2, y2), 255, -1)

            cleaned = image_bgr.copy()
            cleaned[removal_mask > 0] = (
                self.config.cleaning.fill_value,
                self.config.cleaning.fill_value,
                self.config.cleaning.fill_value,
            )

            keep_mask = np.full((height, width), 255, dtype=np.uint8)
            keep_mask[removal_mask > 0] = 0

            cleaned_image = Image.fromarray(cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB))
            keep_mask_image = Image.fromarray(keep_mask, mode="L")
            mask_coverage = float((keep_mask > 0).mean())

            return cleaned_image, keep_mask_image, {
                "removed_regions": float(len(boxes)),
                "mask_coverage": round(mask_coverage, 6),
            }
        except Exception as exc:
            self.logger.error(f"图纸清洗失败，将回退到原图: {exc}", exc_info=True)
            keep_mask = self._default_mask(image)
            return image, keep_mask, {"removed_regions": 0.0, "mask_coverage": 1.0}

    def clean_image(self, image: Image.Image) -> Image.Image:
        cleaned_image, _, _ = self.clean_image_with_mask(image)
        return cleaned_image


_cleaning_service: Optional[CleaningService] = None


def get_cleaning_service() -> CleaningService:
    global _cleaning_service
    if _cleaning_service is None:
        _cleaning_service = CleaningService()
    return _cleaning_service


def init_cleaning_service(model_path: Optional[str] = None):
    global _cleaning_service
    if _cleaning_service is None:
        _cleaning_service = CleaningService()
    _cleaning_service.initialize(model_path)
