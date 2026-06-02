"""
Feature extraction service.

Key improvements:
1. Fix broken import path to the retrieval model module.
2. Add mask-aware feature extraction bundle for retrieval and indexing.
3. Produce lightweight structure descriptors for result re-ranking.
"""
from __future__ import annotations

import asyncio
import time
from queue import Empty, Queue
from threading import Thread
from typing import Dict, List, Optional, Tuple

import clip
import numpy as np
import torch
from PIL import Image

from config.config_manager import get_config
from module.retrieval_model import load_pretrained_model
from module.structure_features import build_structure_descriptor
from monitoring.logger import get_logger
from monitoring.metrics import get_metrics_collector
from services.cleaning_service import get_cleaning_service
from services.ocr_service import extract_title_block_fields, get_ocr_service


class FeatureService:
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger("feature_service")
        self.metrics = get_metrics_collector()

        self.device = self.config.model.device
        self.model = None
        self.preprocess = None
        self.class_names = None
        self.metadata = None

        self.async_enabled = self.config.feature.async_enabled
        self.task_queue = Queue(maxsize=self.config.feature.max_queue_size)
        self.worker_thread = None
        self._running = False
        self._text_embedding_cache: Dict[str, List[float]] = {}

        self._load_model()
        if self.async_enabled:
            self._start_worker()

    def _load_model(self):
        try:
            self.logger.info(f"开始加载模型: {self.config.model.finetuned_model_path}")
            self.logger.info(f"使用设备: {self.device}")

            self.model, self.class_names, self.metadata = load_pretrained_model(
                self.config.model.finetuned_model_path,
                device=self.device,
            )

            _, self.preprocess = clip.load(
                self.config.model.clip_model_name,
                device=self.device,
            )
            self.model.eval()
            self.logger.info("模型加载完成")
        except Exception as exc:
            self.logger.error(f"模型加载失败: {exc}", exc_info=True)
            raise

    def _start_worker(self):
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self._running = True
            self.worker_thread = Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
            self.logger.info("异步特征提取线程已启动")

    def _worker_loop(self):
        while self._running:
            try:
                task = self.task_queue.get(timeout=1.0)
                if task is None:
                    break
                image, callback = task
                started_at = time.time()
                try:
                    bundle = self.extract_feature_bundle(image)
                    callback(bundle, True)
                    self.metrics.record_feature_extraction(time.time() - started_at, True)
                except Exception as exc:
                    self.logger.error(f"异步特征提取失败: {exc}", exc_info=True)
                    callback(exc, False)
                    self.metrics.record_feature_extraction(time.time() - started_at, False)
                self.task_queue.task_done()
            except Empty:
                continue
            except Exception as exc:
                self.logger.error(f"特征线程异常: {exc}", exc_info=True)

    def _fallback_keep_mask(self, image: Image.Image) -> Image.Image:
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
        keep = np.where(gray < 245, 255, 0).astype(np.uint8)
        if keep.max() == 0:
            keep[:] = 255
        return Image.fromarray(keep, mode="L")

    def _apply_cleaning(self, image: Image.Image):
        cleaner = get_cleaning_service()
        cleaned_image, mask_image, meta = cleaner.clean_image_with_mask(image)

        if mask_image is None:
            mask_image = self._fallback_keep_mask(cleaned_image)
        else:
            mask_array = np.asarray(mask_image.convert("L"), dtype=np.uint8)
            if mask_array.max() == 0:
                mask_image = self._fallback_keep_mask(cleaned_image)

        return cleaned_image, mask_image, meta

    def _build_patch_mask(self, mask_image: Optional[Image.Image]) -> Optional[torch.Tensor]:
        if mask_image is None:
            return None

        input_resolution = getattr(self.model.clip_model.visual, "input_resolution", 224)
        patch_size = self.model.get_patch_size() if hasattr(self.model, "get_patch_size") else 32
        grid_size = max(1, input_resolution // patch_size)

        resized_mask = mask_image.convert("L").resize(
            (input_resolution, input_resolution),
            Image.Resampling.NEAREST,
        )
        mask_array = np.asarray(resized_mask, dtype=np.float32) / 255.0
        patch_mask = mask_array.reshape(grid_size, patch_size, grid_size, patch_size).mean(axis=(1, 3))
        patch_mask = (patch_mask > 0.1).astype(np.float32).reshape(1, -1)
        if patch_mask.max() == 0:
            patch_mask[:] = 1.0
        return torch.from_numpy(patch_mask).to(self.device)

    def encode_text_content(self, text: str) -> Optional[List[float]]:
        normalized = " ".join(str(text or "").split())[: self.config.innovation.ocr_max_text_length]
        if not normalized:
            return None
        cached = self._text_embedding_cache.get(normalized)
        if cached is not None:
            return cached

        tokens = clip.tokenize([normalized], truncate=True).to(self.device)
        with torch.no_grad():
            text_feature = self.model.clip_model.encode_text(tokens)
            text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        embedding = text_feature.cpu().numpy().flatten().tolist()
        self._text_embedding_cache[normalized] = embedding
        return embedding

    def _empty_text_descriptor(self) -> Dict:
        return {
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
            "title_block_fields": extract_title_block_fields([]),
        }

    def _resolve_feature_flags(
        self,
        use_cleaning: Optional[bool] = None,
        use_masked_pooling: Optional[bool] = None,
        use_ocr_text: Optional[bool] = None,
    ) -> Tuple[bool, bool, bool]:
        resolved_cleaning = self.config.cleaning.enabled if use_cleaning is None else bool(use_cleaning)
        resolved_masked_pooling = (
            self.config.innovation.enable_masked_pooling
            if use_masked_pooling is None
            else bool(use_masked_pooling)
        )
        resolved_ocr_text = (
            self.config.innovation.enable_ocr_text_fusion
            if use_ocr_text is None
            else bool(use_ocr_text)
        )
        return resolved_cleaning, resolved_masked_pooling, resolved_ocr_text

    def _prepare_feature_inputs(
        self,
        image: Image.Image,
        use_cleaning: bool,
        use_masked_pooling: bool,
    ) -> Dict:
        if use_cleaning:
            cleaned_image, mask_image, cleaning_meta = self._apply_cleaning(image)
        else:
            cleaned_image = image.convert("RGB")
            mask_image = self._fallback_keep_mask(cleaned_image)
            cleaning_meta = {"removed_regions": 0.0, "mask_coverage": 1.0}

        patch_mask = None
        if use_masked_pooling or self.config.innovation.adapter_type == "spatial_attention":
            patch_mask = self._build_patch_mask(mask_image)

        tensor = self.preprocess(cleaned_image)
        return {
            "source_image": image,
            "cleaned_image": cleaned_image,
            "mask_image": mask_image,
            "patch_mask": patch_mask,
            "tensor": tensor,
            "cleaning_meta": cleaning_meta,
        }

    def _encode_embeddings_batch(self, prepared_items: List[Dict], use_masked_pooling: bool) -> List[List[float]]:
        if not prepared_items:
            return []

        batch_tensor = torch.stack([item["tensor"] for item in prepared_items], dim=0).to(self.device)
        batch_mask = None
        if prepared_items[0]["patch_mask"] is not None:
            batch_mask = torch.cat([item["patch_mask"] for item in prepared_items], dim=0).to(self.device)

        with torch.no_grad():
            features = self.model(
                batch_tensor,
                return_features=True,
                mask=(
                    batch_mask
                    if use_masked_pooling or self.config.innovation.adapter_type == "spatial_attention"
                    else None
                ),
            )
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().tolist()

    def extract_feature_bundles_batch(
        self,
        images: List[Image.Image],
        use_cleaning: Optional[bool] = None,
        use_masked_pooling: Optional[bool] = None,
        use_ocr_text: Optional[bool] = None,
    ) -> List[Dict]:
        if not images:
            return []

        started_at = time.time()
        use_cleaning, use_masked_pooling, use_ocr_text = self._resolve_feature_flags(
            use_cleaning=use_cleaning,
            use_masked_pooling=use_masked_pooling,
            use_ocr_text=use_ocr_text,
        )
        prepared_items = [
            self._prepare_feature_inputs(image, use_cleaning=use_cleaning, use_masked_pooling=use_masked_pooling)
            for image in images
        ]
        embeddings = self._encode_embeddings_batch(prepared_items, use_masked_pooling=use_masked_pooling)
        total_duration = time.time() - started_at
        duration_per_image = total_duration / max(1, len(images))
        bundles: List[Dict] = []

        for prepared, embedding in zip(prepared_items, embeddings):
            cleaned_image = prepared["cleaned_image"]
            mask_image = prepared["mask_image"]
            text_descriptor = (
                get_ocr_service().extract_text_descriptor(prepared["source_image"], mask_image=mask_image)
                if use_ocr_text
                else self._empty_text_descriptor()
            )
            text_embedding = None
            if use_ocr_text and text_descriptor.get("has_text"):
                text_embedding = self.encode_text_content(text_descriptor.get("raw_text", ""))

            bundles.append(
                {
                    "embedding": embedding,
                    "cleaned_image": cleaned_image,
                    "mask_image": mask_image,
                    "patch_mask": (
                        prepared["patch_mask"].cpu().numpy().flatten().tolist()
                        if prepared["patch_mask"] is not None
                        else None
                    ),
                    "structure_descriptor": build_structure_descriptor(
                        cleaned_image,
                        mask_image=mask_image,
                        grid_size=self.config.innovation.descriptor_grid_size,
                    ),
                    "text_descriptor": text_descriptor,
                    "text_embedding": text_embedding,
                    "cleaning_meta": prepared["cleaning_meta"],
                    "duration_seconds": round(duration_per_image, 6),
                    "experiment_flags": {
                        "use_cleaning": use_cleaning,
                        "use_masked_pooling": use_masked_pooling,
                        "use_ocr_text": use_ocr_text,
                    },
                }
            )

        self.metrics.record_feature_extraction(total_duration, True)
        return bundles

    def extract_feature_bundle(
        self,
        image: Image.Image,
        use_cleaning: Optional[bool] = None,
        use_masked_pooling: Optional[bool] = None,
        use_ocr_text: Optional[bool] = None,
    ) -> Dict:
        return self.extract_feature_bundles_batch(
            [image],
            use_cleaning=use_cleaning,
            use_masked_pooling=use_masked_pooling,
            use_ocr_text=use_ocr_text,
        )[0]

    def extract_feature(self, image: Image.Image) -> List[float]:
        bundle = self.extract_feature_bundle(image)
        return bundle["embedding"]

    async def extract_feature_async(self, image: Image.Image) -> List[float]:
        if not self.async_enabled:
            return self.extract_feature(image)

        loop = asyncio.get_event_loop()
        future = loop.create_future()

        def callback(payload, success):
            if success:
                loop.call_soon_threadsafe(future.set_result, payload["embedding"])
            else:
                loop.call_soon_threadsafe(future.set_exception, payload)

        self.task_queue.put((image, callback), timeout=5.0)
        return await future

    def extract_features_batch(self, images: List[Image.Image]) -> List[List[float]]:
        bundles = self.extract_feature_bundles_batch(images)
        return [bundle["embedding"] for bundle in bundles]

    def get_model_info(self) -> dict:
        return {
            "device": self.device,
            "model_path": self.config.model.finetuned_model_path,
            "embedding_dim": self.config.model.embedding_dim,
            "class_names": self.class_names,
            "metadata": self.metadata,
            "async_enabled": self.async_enabled,
            "adapter_type": self.config.innovation.adapter_type,
            "enable_masked_pooling": self.config.innovation.enable_masked_pooling,
            "enable_structure_rerank": self.config.innovation.enable_structure_rerank,
            "enable_ocr_text_fusion": self.config.innovation.enable_ocr_text_fusion,
            "ocr_fusion_strategy": self.config.innovation.ocr_fusion_strategy,
        }

    def shutdown(self):
        self._running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.task_queue.put(None)
            self.worker_thread.join(timeout=5.0)
        self.logger.info("特征服务已关闭")


_feature_service: Optional[FeatureService] = None


def get_feature_service() -> FeatureService:
    global _feature_service
    if _feature_service is None:
        _feature_service = FeatureService()
    return _feature_service


def init_feature_service():
    global _feature_service
    _feature_service = FeatureService()
