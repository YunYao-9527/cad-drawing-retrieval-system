"""
Configuration management.

This module keeps the original YAML-based workflow, but adds structured
configuration for the new masked-attention and structure-rerank pipeline.
"""
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator


load_dotenv()


class AppConfig(BaseModel):
    name: str = "CAD图纸检索系统"
    version: str = "2.0.0"
    host: str = "127.0.0.1"
    port: int = 5000
    debug: bool = False
    threaded: bool = True


class ModelConfig(BaseModel):
    finetuned_model_path: str
    clip_model_name: str = "ViT-B/32"
    embedding_dim: int = 512
    device: str = "auto"  # auto, cuda, cpu

    @validator("device")
    def validate_device(cls, value: str) -> str:
        if value == "auto":
            try:
                import torch

                return "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                return "cpu"
        return value


class GalleryConfig(BaseModel):
    cad_drawing_dir: str


class ChromaConfig(BaseModel):
    persist_directory: str = "./chroma_db"


class QdrantConfig(BaseModel):
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "cad_drawings"


class MilvusConfig(BaseModel):
    host: str = "localhost"
    port: int = 19530
    collection_name: str = "cad_drawings"


class VectorDBConfig(BaseModel):
    type: str = "qdrant"  # qdrant only in current project
    persist_directory: str = "./chroma_db"
    collection_name: str = "cad_drawings"
    chroma: ChromaConfig = Field(default_factory=ChromaConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    milvus: MilvusConfig = Field(default_factory=MilvusConfig)


class FeatureConfig(BaseModel):
    batch_size: int = 100
    max_queue_size: int = 1000
    async_enabled: bool = True
    gpu_memory_fraction: float = 0.8


class CleaningConfig(BaseModel):
    enabled: bool = True
    model_path: Optional[str] = None
    confidence: float = 0.25
    iou: float = 0.45
    padding: int = 5
    fill_value: int = 255


class InnovationConfig(BaseModel):
    adapter_type: str = "mlp"  # mlp, spatial_attention
    enable_masked_pooling: bool = True
    enable_structure_rerank: bool = True
    structure_rerank_weight: float = 0.2
    enable_ocr_text_fusion: bool = False
    ocr_fusion_strategy: str = "legacy"  # legacy, hybrid_rerank
    ocr_text_weight: float = 0.15
    ocr_clip_text_weight: float = 0.35
    ocr_bonus_threshold: float = 0.35
    ocr_candidate_multiplier: int = 6
    ocr_rerank_topn: int = 30
    ocr_identifier_bonus: float = 0.08
    ocr_partial_identifier_bonus: float = 0.04
    ocr_min_confidence: float = 0.5
    ocr_max_text_length: int = 500
    ocr_use_mask_regions: bool = True
    descriptor_grid_size: int = 16
    attention_hidden_dim: int = 256
    attention_heads: int = 4
    attention_dropout: float = 0.1

    @validator("adapter_type")
    def validate_adapter_type(cls, value: str) -> str:
        allowed = {"mlp", "spatial_attention"}
        if value not in allowed:
            raise ValueError(f"adapter_type must be one of {sorted(allowed)}")
        return value

    @validator("ocr_fusion_strategy")
    def validate_ocr_fusion_strategy(cls, value: str) -> str:
        allowed = {"legacy", "hybrid_rerank"}
        if value not in allowed:
            raise ValueError(f"ocr_fusion_strategy must be one of {sorted(allowed)}")
        return value

    @validator("structure_rerank_weight")
    def validate_structure_rerank_weight(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("structure_rerank_weight must be between 0 and 1")
        return value

    @validator("ocr_text_weight")
    def validate_ocr_text_weight(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("ocr_text_weight must be between 0 and 1")
        return value

    @validator("ocr_clip_text_weight")
    def validate_ocr_clip_text_weight(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("ocr_clip_text_weight must be between 0 and 1")
        return value

    @validator("ocr_bonus_threshold")
    def validate_ocr_bonus_threshold(cls, value: float) -> float:
        if not 0.0 <= value < 1.0:
            raise ValueError("ocr_bonus_threshold must be between 0 and 1")
        return value


class RetrievalConfig(BaseModel):
    default_top_k: int = 10
    max_top_k: int = 100
    similarity_threshold: float = 0.0
    enable_refinement: bool = True


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "json"  # json, text
    file: str = "logs/app.log"
    max_bytes: int = 10485760
    backup_count: int = 5
    enable_console: bool = True


class MonitoringConfig(BaseModel):
    enable_prometheus: bool = True
    prometheus_port: int = 9090
    metrics_path: str = "/metrics"
    enable_performance_tracking: bool = True


class HealthConfig(BaseModel):
    enable_health_check: bool = True
    health_check_interval: int = 30
    enable_detailed_checks: bool = True


class SystemConfig(BaseModel):
    app: AppConfig
    model: ModelConfig
    gallery: GalleryConfig
    vector_db: VectorDBConfig
    feature: FeatureConfig
    cleaning: CleaningConfig = Field(default_factory=CleaningConfig)
    innovation: InnovationConfig = Field(default_factory=InnovationConfig)
    retrieval: RetrievalConfig
    logging: LoggingConfig
    monitoring: MonitoringConfig
    health: HealthConfig


class ConfigManager:
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        self.config_path = Path(config_path)
        self._config: Optional[SystemConfig] = None
        self._load_config()

    def _load_config(self) -> None:
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as handle:
                yaml_config = yaml.safe_load(handle) or {}
        else:
            yaml_config = {}

        yaml_config = self._apply_env_overrides(yaml_config)
        try:
            self._config = SystemConfig(**yaml_config)
        except Exception as exc:
            raise ValueError(f"配置加载失败: {exc}") from exc

    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        if os.getenv("APP_HOST"):
            config.setdefault("app", {})["host"] = os.getenv("APP_HOST")
        if os.getenv("APP_PORT"):
            config.setdefault("app", {})["port"] = int(os.getenv("APP_PORT"))
        if os.getenv("APP_DEBUG"):
            config.setdefault("app", {})["debug"] = os.getenv("APP_DEBUG").lower() == "true"

        if os.getenv("MODEL_FINETUNED_MODEL_PATH"):
            config.setdefault("model", {})["finetuned_model_path"] = os.getenv("MODEL_FINETUNED_MODEL_PATH")
        if os.getenv("MODEL_DEVICE"):
            config.setdefault("model", {})["device"] = os.getenv("MODEL_DEVICE")

        if os.getenv("GALLERY_CAD_DRAWING_DIR"):
            config.setdefault("gallery", {})["cad_drawing_dir"] = os.getenv("GALLERY_CAD_DRAWING_DIR")

        if os.getenv("CLEANING_ENABLED"):
            config.setdefault("cleaning", {})["enabled"] = os.getenv("CLEANING_ENABLED").lower() == "true"
        if os.getenv("CLEANING_MODEL_PATH"):
            config.setdefault("cleaning", {})["model_path"] = os.getenv("CLEANING_MODEL_PATH")

        if os.getenv("VECTOR_DB_TYPE"):
            config.setdefault("vector_db", {})["type"] = os.getenv("VECTOR_DB_TYPE")
        if os.getenv("VECTOR_DB_PERSIST_DIRECTORY"):
            config.setdefault("vector_db", {})["persist_directory"] = os.getenv("VECTOR_DB_PERSIST_DIRECTORY")

        if os.getenv("INNOVATION_ADAPTER_TYPE"):
            config.setdefault("innovation", {})["adapter_type"] = os.getenv("INNOVATION_ADAPTER_TYPE")
        if os.getenv("INNOVATION_ENABLE_MASKED_POOLING"):
            config.setdefault("innovation", {})["enable_masked_pooling"] = (
                os.getenv("INNOVATION_ENABLE_MASKED_POOLING").lower() == "true"
            )
        if os.getenv("INNOVATION_ENABLE_STRUCTURE_RERANK"):
            config.setdefault("innovation", {})["enable_structure_rerank"] = (
                os.getenv("INNOVATION_ENABLE_STRUCTURE_RERANK").lower() == "true"
            )
        if os.getenv("INNOVATION_STRUCTURE_RERANK_WEIGHT"):
            config.setdefault("innovation", {})["structure_rerank_weight"] = float(
                os.getenv("INNOVATION_STRUCTURE_RERANK_WEIGHT")
            )
        if os.getenv("INNOVATION_ENABLE_OCR_TEXT_FUSION"):
            config.setdefault("innovation", {})["enable_ocr_text_fusion"] = (
                os.getenv("INNOVATION_ENABLE_OCR_TEXT_FUSION").lower() == "true"
            )
        if os.getenv("INNOVATION_OCR_FUSION_STRATEGY"):
            config.setdefault("innovation", {})["ocr_fusion_strategy"] = os.getenv("INNOVATION_OCR_FUSION_STRATEGY")
        if os.getenv("INNOVATION_OCR_TEXT_WEIGHT"):
            config.setdefault("innovation", {})["ocr_text_weight"] = float(
                os.getenv("INNOVATION_OCR_TEXT_WEIGHT")
            )
        if os.getenv("INNOVATION_OCR_BONUS_THRESHOLD"):
            config.setdefault("innovation", {})["ocr_bonus_threshold"] = float(
                os.getenv("INNOVATION_OCR_BONUS_THRESHOLD")
            )
        if os.getenv("INNOVATION_OCR_MIN_CONFIDENCE"):
            config.setdefault("innovation", {})["ocr_min_confidence"] = float(
                os.getenv("INNOVATION_OCR_MIN_CONFIDENCE")
            )
        if os.getenv("INNOVATION_OCR_CLIP_TEXT_WEIGHT"):
            config.setdefault("innovation", {})["ocr_clip_text_weight"] = float(
                os.getenv("INNOVATION_OCR_CLIP_TEXT_WEIGHT")
            )

        if os.getenv("LOGGING_LEVEL"):
            config.setdefault("logging", {})["level"] = os.getenv("LOGGING_LEVEL")

        return config

    def get_config(self) -> SystemConfig:
        if self._config is None:
            self._load_config()
        return self._config

    def reload(self) -> None:
        self._load_config()

    def get(self, key: str, default: Any = None) -> Any:
        value: Any = self._config.dict() if self._config else {}
        for part in key.split("."):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        return value


_config_manager: Optional[ConfigManager] = None


def get_config() -> SystemConfig:
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager.get_config()


def init_config(config_path: Optional[str] = None) -> None:
    global _config_manager
    _config_manager = ConfigManager(config_path)
