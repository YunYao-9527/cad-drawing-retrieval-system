"""Service package with lazy exports to avoid circular imports."""

__all__ = [
    "FeatureService",
    "get_feature_service",
    "init_feature_service",
    "RetrievalService",
    "get_retrieval_service",
    "CleaningService",
    "get_cleaning_service",
    "init_cleaning_service",
    "OCRService",
    "get_ocr_service",
    "init_ocr_service",
]


def __getattr__(name):
    if name in {"FeatureService", "get_feature_service", "init_feature_service"}:
        from .feature_service import FeatureService, get_feature_service, init_feature_service

        return {
            "FeatureService": FeatureService,
            "get_feature_service": get_feature_service,
            "init_feature_service": init_feature_service,
        }[name]

    if name in {"RetrievalService", "get_retrieval_service"}:
        from .retrieval_service import RetrievalService, get_retrieval_service

        return {
            "RetrievalService": RetrievalService,
            "get_retrieval_service": get_retrieval_service,
        }[name]

    if name in {"CleaningService", "get_cleaning_service", "init_cleaning_service"}:
        from .cleaning_service import CleaningService, get_cleaning_service, init_cleaning_service

        return {
            "CleaningService": CleaningService,
            "get_cleaning_service": get_cleaning_service,
            "init_cleaning_service": init_cleaning_service,
        }[name]

    if name in {"OCRService", "get_ocr_service", "init_ocr_service"}:
        from .ocr_service import OCRService, get_ocr_service, init_ocr_service

        return {
            "OCRService": OCRService,
            "get_ocr_service": get_ocr_service,
            "init_ocr_service": init_ocr_service,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
