"""
Qdrant vector database implementation.
"""
from __future__ import annotations

import glob
import hashlib
import os
import sys
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.models import Distance, PointStruct, VectorParams
except ImportError:  # pragma: no cover - optional dependency for import-time resilience
    QdrantClient = None
    models = None
    Distance = None
    PointStruct = None
    VectorParams = None

from config.config_manager import get_config
from module.structure_features import compare_structure_descriptors
from monitoring.logger import get_logger
from monitoring.metrics import get_metrics_collector
from services.ocr_service import (
    analyze_text_descriptors,
    build_filename_descriptor,
    build_text_descriptor,
    extract_title_block_fields,
    compare_title_block_fields,
    compare_text_descriptors,
)
from services.feature_service import get_feature_service


class QdrantVectorDB:
    def __init__(self):
        if QdrantClient is None:
            raise ImportError("qdrant-client is not installed.")

        self.logger = get_logger("qdrant_db")
        self.metrics = get_metrics_collector()
        self.config = get_config()
        self.feature_service = get_feature_service()

        qdrant_config = self.config.vector_db.qdrant
        self.collection_name = self.config.vector_db.collection_name
        self.embedding_dim = self.config.model.embedding_dim

        use_remote = qdrant_config.host not in ["localhost", "127.0.0.1"]
        if use_remote:
            try:
                self.client = QdrantClient(
                    host=qdrant_config.host,
                    port=qdrant_config.port,
                    timeout=3,
                )
                self.client.get_collections()
                self.logger.info(f"Connected to remote Qdrant: {qdrant_config.host}:{qdrant_config.port}")
            except Exception as exc:
                self.logger.warning(f"Remote Qdrant unavailable, falling back to local mode: {exc}")
                use_remote = False

        if not use_remote:
            qdrant_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "qdrant_db")
            os.makedirs(qdrant_path, exist_ok=True)
            self.client = QdrantClient(path=qdrant_path)
            self.logger.info(f"Using local Qdrant: {qdrant_path}")

        self._ensure_collection()

    def _ensure_collection(self):
        collections = self.client.get_collections()
        collection_list = collections.collections if hasattr(collections, "collections") else collections
        exists = any(collection.name == self.collection_name for collection in collection_list)

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE),
            )
            self.logger.info(f"Created collection: {self.collection_name}")

    def reset_collection(self) -> None:
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            self.logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as exc:
            self.logger.warning(f"Failed to delete collection {self.collection_name}: {exc}")
        self._ensure_collection()

    def extract_feature(self, image):
        return self.feature_service.extract_feature(image)

    def extract_class_from_path(self, filepath: str) -> str:
        filepath = filepath.replace("\\", "/")
        relative_path = os.path.relpath(filepath, self.config.gallery.cad_drawing_dir).replace("\\", "/")
        parts = relative_path.split("/")
        if len(parts) > 1:
            return parts[0]
        return os.path.splitext(os.path.basename(filepath))[0]

    def scan_images(self) -> List[str]:
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
        images: List[str] = []
        for ext in image_extensions:
            pattern = os.path.join(self.config.gallery.cad_drawing_dir, "**", ext)
            images.extend(glob.glob(pattern, recursive=True))
        return images

    def _path_to_id(self, rel_path: str) -> int:
        digest = hashlib.md5(rel_path.encode("utf-8")).hexdigest()[:16]
        return int(digest, 16)

    def _existing_rel_paths(self) -> set:
        rel_paths = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                if payload.get("rel_path"):
                    rel_paths.add(payload["rel_path"])
            if offset is None:
                break
        return rel_paths

    def _build_payload(self, full_path: str, rel_path: str, feature_bundle: Dict, image_class: str) -> Dict:
        structure_descriptor = feature_bundle["structure_descriptor"]
        cleaning_meta = feature_bundle.get("cleaning_meta", {})
        text_descriptor = feature_bundle.get("text_descriptor", {})
        title_fields = text_descriptor.get("title_block_fields", {})
        field_sources = title_fields.get("field_sources", {})
        return {
            "filename": os.path.basename(full_path),
            "filepath": full_path,
            "rel_path": rel_path,
            "class": image_class,
            "filesize": os.path.getsize(full_path),
            "added_time": str(np.datetime64("now")),
            "structure_grid": structure_descriptor.get("grid", []),
            "horizontal_profile": structure_descriptor.get("horizontal_profile", []),
            "vertical_profile": structure_descriptor.get("vertical_profile", []),
            "foreground_ratio": float(structure_descriptor.get("foreground_ratio", 0.0)),
            "mask_coverage": float(cleaning_meta.get("mask_coverage", 1.0)),
            "removed_regions": float(cleaning_meta.get("removed_regions", 0.0)),
            "ocr_text": text_descriptor.get("raw_text", ""),
            "ocr_text_lines": text_descriptor.get("text_lines", []),
            "ocr_title_block_lines": text_descriptor.get("title_block_lines", []),
            "ocr_title_block_entries": text_descriptor.get("title_block_entries", []),
            "ocr_tokens": text_descriptor.get("tokens", []),
            "ocr_number_tokens": text_descriptor.get("number_tokens", []),
            "ocr_identifier_tokens": text_descriptor.get("identifier_tokens", []),
            "ocr_chargrams": text_descriptor.get("chargrams", []),
            "ocr_has_text": bool(text_descriptor.get("has_text", False)),
            "ocr_source": text_descriptor.get("source", "disabled"),
            "ocr_text_quality": float(text_descriptor.get("text_quality", 0.0)),
            "ocr_field_part_name": title_fields.get("part_name", ""),
            "ocr_field_drawing_no": title_fields.get("drawing_no", ""),
            "ocr_field_material": title_fields.get("material", ""),
            "ocr_field_scale": title_fields.get("scale", ""),
            "ocr_field_quantity": title_fields.get("quantity", ""),
            "ocr_field_quality": float(title_fields.get("field_quality", 0.0)),
            "ocr_field_kv_pair_count": int(title_fields.get("kv_pair_count", 0)),
            "ocr_field_detection_method": title_fields.get("detection_method", "rules"),
            "ocr_field_sources": field_sources,
            "ocr_text_embedding": feature_bundle.get("text_embedding"),
        }

    def initialize_database(self, batch_size: Optional[int] = None) -> int:
        batch_size = batch_size or self.config.feature.batch_size
        image_files = self.scan_images()
        if not image_files:
            self.logger.warning("No gallery images found, skipping index build.")
            return 0

        existing_rel_paths = self._existing_rel_paths()
        pending_points: List[PointStruct] = []
        processed = 0
        skipped = 0
        use_tqdm = bool(getattr(sys.stderr, "isatty", lambda: False)())

        with tqdm(
            total=len(image_files),
            desc="Building gallery index",
            unit="img",
            ncols=100,
            disable=not use_tqdm,
        ) as progress:
            for chunk_start in range(0, len(image_files), batch_size):
                chunk_files = image_files[chunk_start : chunk_start + batch_size]
                batch_images = []
                batch_records = []

                for full_path in chunk_files:
                    rel_path = os.path.relpath(full_path, self.config.gallery.cad_drawing_dir).replace("\\", "/")
                    if rel_path in existing_rel_paths:
                        skipped += 1
                        progress.update(1)
                        continue

                    try:
                        image = Image.open(full_path).convert("RGB")
                        image_class = self.extract_class_from_path(full_path)
                        batch_images.append(image)
                        batch_records.append((full_path, rel_path, image_class))
                    except Exception as exc:
                        message = f"Skipping file {os.path.basename(full_path)}: {exc}"
                        if use_tqdm:
                            progress.write(message)
                        else:
                            self.logger.warning(message)
                        progress.update(1)

                if not batch_records:
                    continue

                try:
                    feature_bundles = self.feature_service.extract_feature_bundles_batch(batch_images)
                except Exception as exc:
                    message = f"Batch feature extraction failed for chunk starting at {chunk_start}: {exc}"
                    if use_tqdm:
                        progress.write(message)
                    else:
                        self.logger.warning(message)
                    feature_bundles = []
                    for image in batch_images:
                        try:
                            feature_bundles.append(self.feature_service.extract_feature_bundle(image))
                        except Exception as inner_exc:
                            feature_bundles.append(inner_exc)

                for record, feature_bundle in zip(batch_records, feature_bundles):
                    full_path, rel_path, image_class = record
                    try:
                        if isinstance(feature_bundle, Exception):
                            raise feature_bundle
                        payload = self._build_payload(full_path, rel_path, feature_bundle, image_class)
                        point = PointStruct(
                            id=self._path_to_id(rel_path),
                            vector=feature_bundle["embedding"],
                            payload=payload,
                        )
                        pending_points.append(point)
                        processed += 1
                        existing_rel_paths.add(rel_path)

                        if len(pending_points) >= batch_size:
                            self.client.upsert(collection_name=self.collection_name, points=pending_points)
                            pending_points = []
                    except Exception as exc:
                        message = f"Skipping file {os.path.basename(full_path)}: {exc}"
                        if use_tqdm:
                            progress.write(message)
                        else:
                            self.logger.warning(message)
                    finally:
                        progress.update(1)

        if pending_points:
            self.client.upsert(collection_name=self.collection_name, points=pending_points)

        self.logger.info(f"Index build completed: added={processed}, skipped={skipped}")
        return processed

    def add_image(self, image_path: str, image_class: Optional[str] = None) -> bool:
        if not os.path.exists(image_path):
            self.logger.warning(f"File does not exist: {image_path}")
            return False

        rel_path = os.path.relpath(image_path, self.config.gallery.cad_drawing_dir).replace("\\", "/")
        try:
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(key="rel_path", match=models.MatchValue(value=rel_path))]
                ),
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
            if points:
                return True

            image = Image.open(image_path).convert("RGB")
            feature_bundle = self.feature_service.extract_feature_bundle(image)
            image_class = image_class or self.extract_class_from_path(image_path)
            payload = self._build_payload(image_path, rel_path, feature_bundle, image_class)
            point = PointStruct(
                id=self._path_to_id(rel_path),
                vector=feature_bundle["embedding"],
                payload=payload,
            )
            self.client.upsert(collection_name=self.collection_name, points=[point])
            return True
        except Exception as exc:
            self.logger.error(f"Failed to add image: {exc}", exc_info=True)
            return False

    def delete_image(self, image_id: str) -> bool:
        try:
            clean_id = image_id.strip().replace("\\", "/")
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(key="rel_path", match=models.MatchValue(value=clean_id))]
                ),
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
            if points:
                point_id = points[0].id
            else:
                point_id = self._path_to_id(clean_id)

            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=[point_id]),
            )
            return True
        except Exception as exc:
            self.logger.error(f"Failed to delete image: {exc}", exc_info=True)
            return False

    def _payload_to_descriptor(self, payload: Dict) -> Dict:
        return {
            "grid": payload.get("structure_grid", []),
            "horizontal_profile": payload.get("horizontal_profile", []),
            "vertical_profile": payload.get("vertical_profile", []),
            "foreground_ratio": payload.get("foreground_ratio", 0.0),
        }

    def _payload_to_text_descriptor(self, payload: Dict) -> Dict:
        descriptor = {
            "raw_text": payload.get("ocr_text", ""),
            "text_lines": payload.get("ocr_text_lines", []),
            "title_block_lines": payload.get("ocr_title_block_lines", []),
            "title_block_entries": payload.get("ocr_title_block_entries", []),
            "tokens": payload.get("ocr_tokens", []),
            "number_tokens": payload.get("ocr_number_tokens", []),
            "identifier_tokens": payload.get("ocr_identifier_tokens", []),
            "chargrams": payload.get("ocr_chargrams", []),
            "has_text": payload.get("ocr_has_text", False),
            "text_quality": payload.get("ocr_text_quality", 0.0),
            "title_block_fields": {
                "part_name": payload.get("ocr_field_part_name", ""),
                "drawing_no": payload.get("ocr_field_drawing_no", ""),
                "material": payload.get("ocr_field_material", ""),
                "scale": payload.get("ocr_field_scale", ""),
                "quantity": payload.get("ocr_field_quantity", ""),
                "field_quality": payload.get("ocr_field_quality", 0.0),
                "kv_pair_count": payload.get("ocr_field_kv_pair_count", 0),
                "detection_method": payload.get("ocr_field_detection_method", "rules"),
                "field_sources": payload.get("ocr_field_sources", {}),
            },
        }
        needs_rebuild = bool(descriptor["raw_text"]) and (
            not descriptor["tokens"]
            or not descriptor["identifier_tokens"]
            or float(descriptor.get("text_quality", 0.0) or 0.0) <= 0.0
        )
        if needs_rebuild:
            rebuilt = build_text_descriptor(descriptor["text_lines"] or [descriptor["raw_text"]])
            descriptor.update(rebuilt)
        if (
            not descriptor.get("title_block_fields", {}).get("field_quality", 0.0)
            or not (descriptor.get("title_block_fields", {}).get("field_sources") or {})
        ):
            descriptor["title_block_fields"] = extract_title_block_fields(
                descriptor.get("title_block_lines") or descriptor.get("text_lines") or [descriptor.get("raw_text", "")],
                ocr_entries=descriptor.get("title_block_entries"),
            )
        return descriptor

    def _payload_to_text_embedding(self, payload: Dict) -> Optional[List[float]]:
        embedding = payload.get("ocr_text_embedding")
        if embedding:
            return embedding
        raw_text = payload.get("ocr_text", "")
        if not raw_text:
            return None
        return self.feature_service.encode_text_content(raw_text)

    def _payload_to_filename_descriptor(self, payload: Dict) -> Dict:
        return build_filename_descriptor(payload.get("filename", ""))

    def _resolve_text_fusion_strategy(self, requested_strategy: Optional[str]) -> str:
        strategy = requested_strategy or self.config.innovation.ocr_fusion_strategy
        if strategy not in {"legacy", "hybrid_rerank"}:
            return self.config.innovation.ocr_fusion_strategy
        return strategy

    def search_similar(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        query_descriptor: Optional[Dict] = None,
        query_text_descriptor: Optional[Dict] = None,
        query_text_embedding: Optional[List[float]] = None,
        query_filename: Optional[str] = None,
        text_fusion_strategy: Optional[str] = None,
    ) -> List[Dict]:
        db_count = self.get_count()
        if db_count == 0:
            self.logger.warning("Vector database is empty, search skipped.")
            return []

        rerank_enabled = self.config.innovation.enable_structure_rerank and query_descriptor is not None
        resolved_text_strategy = self._resolve_text_fusion_strategy(text_fusion_strategy)
        text_enabled = self.config.innovation.enable_ocr_text_fusion and (
            query_text_descriptor is not None or (resolved_text_strategy == "hybrid_rerank" and bool(query_filename))
        )
        if not text_enabled:
            resolved_text_strategy = "disabled"
        candidate_multiplier = 2
        if text_enabled and resolved_text_strategy == "hybrid_rerank":
            candidate_multiplier = max(2, int(self.config.innovation.ocr_candidate_multiplier))
        elif rerank_enabled or text_enabled:
            candidate_multiplier = 4
        search_limit = min(top_k * candidate_multiplier, db_count)

        try:
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=search_limit,
                with_payload=True,
                with_vectors=False,
            )
        except AttributeError:
            query_response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=search_limit,
                with_payload=True,
                with_vectors=False,
            )
            search_results = getattr(query_response, "points", [])

        if not search_results:
            return []

        formatted_results = []
        structure_weight = self.config.innovation.structure_rerank_weight if rerank_enabled else 0.0
        text_weight = self.config.innovation.ocr_text_weight if text_enabled else 0.0
        clip_text_weight = self.config.innovation.ocr_clip_text_weight if text_enabled else 0.0
        semantic_weight = max(0.0, 1.0 - structure_weight)
        text_bonus_threshold = self.config.innovation.ocr_bonus_threshold if text_enabled else 1.0
        rerank_topn = min(len(search_results), max(top_k, self.config.innovation.ocr_rerank_topn)) if text_enabled else 0
        query_filename_descriptor = build_filename_descriptor(query_filename or "") if text_enabled else None
        query_title_fields = (
            (query_text_descriptor or {}).get("title_block_fields")
            if text_enabled
            else None
        )
        query_filename_fields = (
            (query_filename_descriptor or {}).get("title_block_fields")
            if text_enabled
            else None
        )

        for point_index, point in enumerate(search_results, start=1):
            payload = point.payload or {}
            cosine_similarity = float(getattr(point, "score", 0.0))
            cosine_similarity = max(-1.0, min(1.0, cosine_similarity))
            distance = float(np.sqrt(max(0.0, 2.0 - 2.0 * cosine_similarity)))
            semantic_similarity = float(max(0.0, min(1.0, (cosine_similarity + 1.0) / 2.0)))

            structure_similarity = 0.0
            if rerank_enabled:
                structure_similarity = compare_structure_descriptors(
                    query_descriptor,
                    self._payload_to_descriptor(payload),
                )
            target_text_descriptor = self._payload_to_text_descriptor(payload) if text_enabled else None
            text_similarity = 0.0
            text_embedding_similarity = 0.0
            text_bonus = 0.0
            identifier_bonus = 0.0
            partial_identifier_bonus = 0.0
            filename_similarity = 0.0
            filename_identifier_bonus = 0.0
            field_similarity = 0.0
            field_bonus = 0.0
            text_analysis = {
                "fusion_score": 0.0,
                "token_similarity": 0.0,
                "number_similarity": 0.0,
                "identifier_similarity": 0.0,
                "chargram_similarity": 0.0,
                "sequence_similarity": 0.0,
                "line_similarity": 0.0,
                "exact_identifier_match": 0.0,
                "partial_identifier_match": 0.0,
                "quality": 0.0,
            }
            field_analysis = {
                "field_score": 0.0,
                "drawing_no_similarity": 0.0,
                "part_name_similarity": 0.0,
                "material_similarity": 0.0,
                "scale_similarity": 0.0,
                "quantity_similarity": 0.0,
                "exact_drawing_no_match": 0.0,
                "field_quality": 0.0,
                "drawing_no_trusted": 0.0,
                "part_name_trusted": 0.0,
                "trusted_field_coverage": 0.0,
            }
            filename_field_analysis = dict(field_analysis)
            if text_enabled:
                text_similarity = compare_text_descriptors(query_text_descriptor, target_text_descriptor)
                text_analysis = analyze_text_descriptors(query_text_descriptor, target_text_descriptor)
                field_analysis = compare_title_block_fields(
                    query_title_fields,
                    target_text_descriptor.get("title_block_fields"),
                )
                filename_field_analysis = compare_title_block_fields(
                    query_filename_fields,
                    self._payload_to_filename_descriptor(payload).get("title_block_fields"),
                )
                field_similarity = max(
                    float(field_analysis.get("field_score", 0.0)),
                    0.85 * float(filename_field_analysis.get("field_score", 0.0)),
                )

            final_similarity = semantic_weight * semantic_similarity + structure_weight * structure_similarity
            if text_enabled and point_index <= rerank_topn:
                if resolved_text_strategy == "hybrid_rerank":
                    target_text_embedding = self._payload_to_text_embedding(payload)
                    filename_analysis = analyze_text_descriptors(
                        query_filename_descriptor,
                        self._payload_to_filename_descriptor(payload),
                    )
                    filename_similarity = float(filename_analysis.get("fusion_score", 0.0))
                    if query_text_embedding and target_text_embedding:
                        text_embedding_similarity = max(
                            0.0,
                            min(1.0, (self.cosine_similarity(query_text_embedding, target_text_embedding) + 1.0) / 2.0),
                        )
                    lexical_anchor = max(
                        text_similarity,
                        filename_similarity,
                        float(text_analysis.get("identifier_similarity", 0.0)),
                        float(filename_analysis.get("identifier_similarity", 0.0)),
                    )
                    combined_text_similarity = (
                        0.75 * text_similarity
                        + 0.25 * filename_similarity
                    )
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
                    if reliable_gain > text_bonus_threshold:
                        normalized_bonus = (reliable_gain - text_bonus_threshold) / max(1e-6, 1.0 - text_bonus_threshold)
                        text_bonus = text_weight * normalized_bonus
                    if text_analysis.get("exact_identifier_match", 0.0) > 0:
                        identifier_bonus = self.config.innovation.ocr_identifier_bonus * float(
                            text_analysis.get("quality", 0.0)
                        )
                    elif text_analysis.get("partial_identifier_match", 0.0) >= 0.92:
                        partial_identifier_bonus = self.config.innovation.ocr_partial_identifier_bonus * float(
                            text_analysis.get("quality", 0.0)
                        )
                    if filename_analysis.get("exact_identifier_match", 0.0) > 0:
                        filename_identifier_bonus = 0.5 * self.config.innovation.ocr_identifier_bonus
                    elif filename_analysis.get("partial_identifier_match", 0.0) >= 0.92:
                        filename_identifier_bonus = 0.5 * self.config.innovation.ocr_partial_identifier_bonus
                elif (
                    text_similarity > text_bonus_threshold
                    and float(text_analysis.get("quality", 0.0)) >= 0.35
                    and (
                        float(text_analysis.get("identifier_similarity", 0.0)) >= 0.05
                        or float(text_analysis.get("exact_identifier_match", 0.0)) > 0
                        or float(text_analysis.get("line_similarity", 0.0)) >= 0.65
                    )
                ):
                    normalized_bonus = (text_similarity - text_bonus_threshold) / max(1e-6, 1.0 - text_bonus_threshold)
                    text_bonus = text_weight * normalized_bonus

                if resolved_text_strategy == "legacy":
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
                        field_bonus = 0.7 * text_weight * field_similarity * quality_gate
                    if (
                        field_analysis.get("exact_drawing_no_match", 0.0) > 0
                        or filename_field_analysis.get("exact_drawing_no_match", 0.0) > 0
                    ) and field_analysis.get("drawing_no_trusted", 0.0) > 0:
                        field_bonus += 0.75 * self.config.innovation.ocr_identifier_bonus
                    elif (
                        field_analysis.get("drawing_no_similarity", 0.0) >= 0.92
                        or filename_field_analysis.get("drawing_no_similarity", 0.0) >= 0.92
                    ) and field_analysis.get("drawing_no_trusted", 0.0) > 0:
                        field_bonus += 0.5 * self.config.innovation.ocr_partial_identifier_bonus
                    if (
                        field_analysis.get("part_name_similarity", 0.0) >= 0.97
                        and field_analysis.get("part_name_trusted", 0.0) > 0
                    ):
                        field_bonus += 0.35 * self.config.innovation.ocr_identifier_bonus
                    elif (
                        field_analysis.get("part_name_similarity", 0.0) >= 0.90
                        and field_analysis.get("part_name_trusted", 0.0) > 0
                    ):
                        field_bonus += 0.2 * self.config.innovation.ocr_partial_identifier_bonus
                    if (
                        field_analysis.get("material_similarity", 0.0) >= 0.98
                        and field_analysis.get("scale_similarity", 0.0) >= 0.98
                    ):
                        field_bonus += 0.1 * self.config.innovation.ocr_identifier_bonus

                final_similarity += (
                    text_bonus
                    + field_bonus
                    + identifier_bonus
                    + partial_identifier_bonus
                    + filename_identifier_bonus
                )
            final_similarity = max(0.0, min(1.0, final_similarity))
            image_id = payload.get("rel_path", str(getattr(point, "id", "")))
            formatted_results.append(
                {
                    "id": image_id,
                    "filename": payload.get("filename", ""),
                    "filepath": payload.get("filepath", ""),
                    "class": payload.get("class", ""),
                    "similarity": float(final_similarity),
                    "semantic_similarity": float(semantic_similarity),
                    "structure_similarity": float(structure_similarity),
                    "text_similarity": float(text_similarity),
                    "text_embedding_similarity": float(text_embedding_similarity),
                    "filename_similarity": float(filename_similarity),
                    "field_similarity": float(field_similarity),
                    "text_bonus": float(text_bonus),
                    "field_bonus": float(field_bonus),
                    "identifier_bonus": float(identifier_bonus),
                    "partial_identifier_bonus": float(partial_identifier_bonus),
                    "filename_identifier_bonus": float(filename_identifier_bonus),
                    "identifier_similarity": float(text_analysis.get("identifier_similarity", 0.0)),
                    "exact_identifier_match": float(text_analysis.get("exact_identifier_match", 0.0)),
                    "partial_identifier_match": float(text_analysis.get("partial_identifier_match", 0.0)),
                    "text_quality": float(text_analysis.get("quality", 0.0)),
                    "field_quality": float(field_analysis.get("field_quality", 0.0)),
                    "drawing_no_similarity": float(field_analysis.get("drawing_no_similarity", 0.0)),
                    "part_name_similarity": float(field_analysis.get("part_name_similarity", 0.0)),
                    "material_similarity": float(field_analysis.get("material_similarity", 0.0)),
                    "scale_similarity": float(field_analysis.get("scale_similarity", 0.0)),
                    "filename_field_similarity": float(filename_field_analysis.get("field_score", 0.0)),
                    "text_fusion_strategy": resolved_text_strategy,
                    "ocr_text": payload.get("ocr_text", ""),
                    "distance": distance,
                    "exists": os.path.exists(payload.get("filepath", "")),
                }
            )

        formatted_results.sort(key=lambda item: item["similarity"], reverse=True)
        formatted_results = formatted_results[:top_k]
        for index, item in enumerate(formatted_results, start=1):
            item["rank"] = index

        return formatted_results

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        array1 = np.asarray(vec1, dtype=np.float32)
        array2 = np.asarray(vec2, dtype=np.float32)
        denom = np.linalg.norm(array1) * np.linalg.norm(array2)
        if denom == 0:
            return 0.0
        return float(np.dot(array1, array2) / denom)

    def refine_similar_results(self, results: List[Dict], query_embedding: List[float]) -> List[Dict]:
        if len(results) < 2:
            return results
        results.sort(key=lambda item: item.get("similarity", 0), reverse=True)
        for index, result in enumerate(results, start=1):
            result["rank"] = index
        return results

    def get_stats(self) -> Dict:
        try:
            count = self.get_count()
            sample_points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=min(1000, max(count, 1)),
                with_payload=True,
                with_vectors=False,
            )
            classes: Dict[str, int] = {}
            for point in sample_points:
                payload = point.payload or {}
                image_class = payload.get("class", "unknown")
                classes[image_class] = classes.get(image_class, 0) + 1

            return {
                "total_images": count,
                "class_distribution": classes,
                "database_path": f"Qdrant: {self.config.vector_db.qdrant.host}:{self.config.vector_db.qdrant.port}",
                "db_type": "qdrant",
                "adapter_type": self.config.innovation.adapter_type,
                "enable_masked_pooling": self.config.innovation.enable_masked_pooling,
                "enable_structure_rerank": self.config.innovation.enable_structure_rerank,
                "enable_ocr_text_fusion": self.config.innovation.enable_ocr_text_fusion,
            }
        except Exception as exc:
            self.logger.error(f"Failed to get database stats: {exc}", exc_info=True)
            return {
                "total_images": 0,
                "class_distribution": {},
                "database_path": "N/A",
                "db_type": "qdrant",
            }

    def get_all_images(self, limit: Optional[int] = None) -> List[Dict]:
        try:
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit or 10000,
                with_payload=True,
                with_vectors=False,
            )
            images = []
            for point in points:
                payload = point.payload or {}
                filepath = payload.get("filepath", "")
                image_id = payload.get("rel_path", str(point.id))
                images.append(
                    {
                        "id": image_id,
                        "filename": payload.get("filename", ""),
                        "filepath": filepath,
                        "class": payload.get("class", ""),
                        "filesize": payload.get("filesize", 0),
                        "added_time": payload.get("added_time", "unknown"),
                        "exists": os.path.exists(filepath) if filepath else False,
                        "mask_coverage": payload.get("mask_coverage", 1.0),
                        "foreground_ratio": payload.get("foreground_ratio", 0.0),
                        "ocr_has_text": payload.get("ocr_has_text", False),
                    }
                )
            return images
        except Exception as exc:
            self.logger.error(f"Failed to list images: {exc}", exc_info=True)
            return []

    def get_count(self) -> int:
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return int(getattr(collection_info, "points_count", 0) or 0)
        except Exception as exc:
            self.logger.warning(f"Failed to get collection count: {exc}")
            return 0

    def cleanup_duplicates(self) -> int:
        try:
            images = self.get_all_images()
            seen = set()
            duplicates = []
            for image in images:
                key = (image["filename"], image["class"])
                if key in seen:
                    duplicates.append(image)
                else:
                    seen.add(key)

            deleted = 0
            for duplicate in duplicates:
                if self.delete_image(duplicate["id"]):
                    deleted += 1
            return deleted
        except Exception as exc:
            self.logger.error(f"Failed to cleanup duplicates: {exc}", exc_info=True)
            return 0

    def find_duplicates(self) -> Dict:
        try:
            images = self.get_all_images()
            groups: Dict = {}
            for image in images:
                key = (image["filename"], image["class"])
                groups.setdefault(key, []).append(image)
            return {key: value for key, value in groups.items() if len(value) > 1}
        except Exception as exc:
            self.logger.error(f"Failed to find duplicates: {exc}", exc_info=True)
            return {}

    def close(self) -> None:
        try:
            close = getattr(self.client, "close", None)
            if callable(close):
                close()
        except Exception as exc:
            self.logger.warning(f"Failed to close Qdrant client cleanly: {exc}")
