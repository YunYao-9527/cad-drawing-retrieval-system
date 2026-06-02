"""
Unified vector database interface.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from config.config_manager import get_config
from database.qdrant_db import QdrantVectorDB
from monitoring.logger import get_logger


class VectorDB:
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger("vector_db")
        self.db_type = self.config.vector_db.type.lower()

        if self.db_type != "qdrant":
            raise ValueError(f"不支持的向量数据库类型: {self.db_type}，当前仅支持 qdrant")

        self.qdrant_db = QdrantVectorDB()
        self.logger.info("使用 Qdrant 向量数据库")

    def extract_feature(self, image):
        return self.qdrant_db.extract_feature(image)

    def extract_class_from_path(self, filepath: str) -> str:
        return self.qdrant_db.extract_class_from_path(filepath)

    def scan_images(self) -> List[str]:
        return self.qdrant_db.scan_images()

    def initialize_database(self, batch_size: Optional[int] = None) -> int:
        return self.qdrant_db.initialize_database(batch_size)

    def reset_collection(self) -> None:
        self.qdrant_db.reset_collection()

    def add_image(self, image_path: str, image_class: Optional[str] = None) -> bool:
        return self.qdrant_db.add_image(image_path, image_class)

    def delete_image(self, image_id: str) -> bool:
        return self.qdrant_db.delete_image(image_id)

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
        return self.qdrant_db.search_similar(
            query_embedding,
            top_k=top_k,
            query_descriptor=query_descriptor,
            query_text_descriptor=query_text_descriptor,
            query_text_embedding=query_text_embedding,
            query_filename=query_filename,
            text_fusion_strategy=text_fusion_strategy,
        )

    def refine_similar_results(self, results: List[Dict], query_embedding: List[float]) -> List[Dict]:
        return self.qdrant_db.refine_similar_results(results, query_embedding)

    def get_stats(self) -> Dict:
        return self.qdrant_db.get_stats()

    def get_all_images(self, limit: Optional[int] = None) -> List[Dict]:
        return self.qdrant_db.get_all_images(limit)

    def cleanup_duplicates(self) -> int:
        return self.qdrant_db.cleanup_duplicates()

    def find_duplicates(self) -> Dict:
        return self.qdrant_db.find_duplicates()

    def close(self) -> None:
        self.qdrant_db.close()


_vector_db: Optional[VectorDB] = None


def init_vector_db():
    global _vector_db
    if _vector_db is None:
        _vector_db = VectorDB()
    return _vector_db


def get_vector_db() -> VectorDB:
    global _vector_db
    if _vector_db is None:
        _vector_db = VectorDB()
    return _vector_db
