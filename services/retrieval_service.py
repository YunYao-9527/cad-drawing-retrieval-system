"""
Retrieval orchestration service.

The service now forwards optional structure descriptors so the vector layer can
perform structure-aware re-ranking on top of semantic retrieval.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

from config import get_config
from database.vector_db import get_vector_db
from monitoring.logger import get_logger
from monitoring.metrics import get_metrics_collector


class RetrievalService:
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger("retrieval_service")
        self.metrics = get_metrics_collector()
        self.vector_db = get_vector_db()

    def search(
        self,
        query_embedding: List[float],
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        category_filter: Optional[str] = None,
        query_descriptor: Optional[Dict] = None,
        enable_structure_rerank: Optional[bool] = None,
        query_text_descriptor: Optional[Dict] = None,
        query_text_embedding: Optional[List[float]] = None,
        query_filename: Optional[str] = None,
        enable_text_fusion: Optional[bool] = None,
        text_fusion_strategy: Optional[str] = None,
    ) -> List[Dict]:
        started_at = time.time()
        try:
            top_k = top_k if top_k is not None else self.config.retrieval.default_top_k
            similarity_threshold = (
                similarity_threshold
                if similarity_threshold is not None
                else self.config.retrieval.similarity_threshold
            )
            top_k = min(top_k, self.config.retrieval.max_top_k)
            if enable_structure_rerank is False:
                query_descriptor = None
            if enable_text_fusion is False:
                query_text_descriptor = None
                query_text_embedding = None

            results = self.vector_db.search_similar(
                query_embedding,
                top_k=top_k,
                query_descriptor=query_descriptor,
                query_text_descriptor=query_text_descriptor,
                query_text_embedding=query_text_embedding,
                query_filename=query_filename,
                text_fusion_strategy=text_fusion_strategy,
            )
            if similarity_threshold > 0 or category_filter:
                results = self._filter_results(results, similarity_threshold, category_filter)

            duration = time.time() - started_at
            self.metrics.record_retrieval(duration, len(results), True)
            self.logger.info(f"检索完成，返回 {len(results)} 条结果，耗时 {duration:.3f}s")
            return results
        except Exception as exc:
            duration = time.time() - started_at
            self.metrics.record_retrieval(duration, 0, False)
            self.logger.error(f"检索失败: {exc}", exc_info=True)
            raise

    def _filter_results(
        self,
        results: List[Dict],
        similarity_threshold: float,
        category_filter: Optional[str],
    ) -> List[Dict]:
        filtered = []
        for result in results:
            if similarity_threshold > 0 and result.get("similarity", 0) < similarity_threshold:
                continue
            if category_filter and result.get("class") != category_filter:
                continue
            filtered.append(result)

        filtered.sort(key=lambda item: item.get("similarity", 0), reverse=True)
        for index, item in enumerate(filtered, start=1):
            item["rank"] = index
        return filtered


_retrieval_service: Optional[RetrievalService] = None


def get_retrieval_service() -> RetrievalService:
    global _retrieval_service
    if _retrieval_service is None:
        _retrieval_service = RetrievalService()
    return _retrieval_service
