"""Module package with lazy exports to avoid importing heavy ML deps eagerly."""

__all__ = [
    "RetrievalCLIPAdapter",
    "ArcFaceLoss",
    "create_retrieval_model",
    "load_pretrained_model",
    "build_structure_descriptor",
    "compare_structure_descriptors",
]


def __getattr__(name):
    if name in {"RetrievalCLIPAdapter", "ArcFaceLoss", "create_retrieval_model", "load_pretrained_model"}:
        from .retrieval_model import ArcFaceLoss, RetrievalCLIPAdapter, create_retrieval_model, load_pretrained_model

        return {
            "RetrievalCLIPAdapter": RetrievalCLIPAdapter,
            "ArcFaceLoss": ArcFaceLoss,
            "create_retrieval_model": create_retrieval_model,
            "load_pretrained_model": load_pretrained_model,
        }[name]

    if name in {"build_structure_descriptor", "compare_structure_descriptors"}:
        from .structure_features import build_structure_descriptor, compare_structure_descriptors

        return {
            "build_structure_descriptor": build_structure_descriptor,
            "compare_structure_descriptors": compare_structure_descriptors,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
