"""
Retrieval model definitions.

This module keeps backward compatibility with the original MLP adapter while
adding two improvements:
1. Spatial-attention adapter for patch-token level modeling.
2. Mask-guided token pooling that can be enabled even with legacy checkpoints.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttentionAdapter(nn.Module):
    """Patch-token attention adapter with optional mask guidance."""

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 512,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.project_down = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.project_up = nn.Linear(hidden_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.gate = nn.Parameter(torch.zeros(1))
        self.classifier = nn.Linear(input_dim, embedding_dim)

    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        patches = tokens[:, 1:, :]
        cls_token = tokens[:, 0, :]

        hidden = self.project_down(patches)
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = (mask <= 0).bool()

        attended, _ = self.attention(hidden, hidden, hidden, key_padding_mask=key_padding_mask)
        fused = patches + self.gate * self.project_up(attended)
        fused = self.layer_norm(fused)

        if mask is not None:
            weights = mask.float().unsqueeze(-1)
            denom = weights.sum(dim=1).clamp_min(1e-6)
            pooled = (fused * weights).sum(dim=1) / denom
        else:
            pooled = fused.mean(dim=1)

        final_feature = pooled + cls_token
        return self.classifier(final_feature)


class RetrievalCLIPAdapter(nn.Module):
    def __init__(
        self,
        clip_model,
        num_classes: Optional[int] = None,
        embedding_dim: int = 512,
        unfreeze_strategy: bool = True,
        adapter_type: str = "mlp",
        attention_hidden_dim: int = 256,
        attention_heads: int = 4,
        attention_dropout: float = 0.1,
        enable_masked_pooling: bool = True,
    ):
        super().__init__()
        self.clip_model = clip_model.float()
        self.feature_dim = clip_model.visual.output_dim
        self.embedding_dim = embedding_dim
        self.unfreeze_strategy = unfreeze_strategy
        self.adapter_type = adapter_type
        self.enable_masked_pooling = enable_masked_pooling

        if adapter_type == "spatial_attention":
            self.adapter = SpatialAttentionAdapter(
                input_dim=self.feature_dim,
                embedding_dim=embedding_dim,
                hidden_dim=attention_hidden_dim,
                num_heads=attention_heads,
                dropout=attention_dropout,
            )
        else:
            self.adapter = nn.Sequential(
                nn.Linear(self.feature_dim, 1024),
                nn.BatchNorm1d(1024),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(512, embedding_dim),
            )

        self.classifier = nn.Linear(embedding_dim, num_classes) if num_classes else None

        if not self.unfreeze_strategy:
            for param in self.clip_model.parameters():
                param.requires_grad = False

    def _clip_grad_enabled(self) -> bool:
        return any(param.requires_grad for param in self.clip_model.parameters())

    def get_patch_size(self) -> int:
        visual = self.clip_model.visual
        if hasattr(visual, "conv1") and hasattr(visual.conv1, "kernel_size"):
            return int(visual.conv1.kernel_size[0])
        return 32

    def _encode_visual_tokens(self, image: torch.Tensor) -> Optional[torch.Tensor]:
        visual = self.clip_model.visual
        if not hasattr(visual, "transformer"):
            return None

        x = visual.conv1(image)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

        class_embedding = visual.class_embedding.to(x.dtype)
        class_embedding = class_embedding + torch.zeros(
            x.shape[0],
            1,
            x.shape[-1],
            dtype=x.dtype,
            device=x.device,
        )
        x = torch.cat([class_embedding, x], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)
        x = visual.ln_post(x)

        if visual.proj is not None:
            x = x @ visual.proj
        return x

    def _encode_base_features(
        self,
        image: torch.Tensor,
        need_tokens: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        grad_enabled = self._clip_grad_enabled()
        if need_tokens:
            context = torch.enable_grad() if grad_enabled else torch.no_grad()
            with context:
                tokens = self._encode_visual_tokens(image)
                if tokens is None:
                    base = self.clip_model.encode_image(image)
                    return base, None
                return tokens[:, 0, :], tokens

        context = torch.enable_grad() if grad_enabled else torch.no_grad()
        with context:
            base = self.clip_model.encode_image(image)
        return base, None

    def _masked_pool(self, tokens: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if tokens is None or mask is None:
            return tokens[:, 0, :] if tokens is not None else None

        patches = tokens[:, 1:, :]
        if patches.shape[1] != mask.shape[1]:
            return tokens[:, 0, :]

        weights = mask.float().unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1e-6)
        pooled = (patches * weights).sum(dim=1) / denom
        return 0.5 * pooled + 0.5 * tokens[:, 0, :]

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x.float()
        need_tokens = self.adapter_type == "spatial_attention" or (
            mask is not None and self.enable_masked_pooling
        )

        base_features, tokens = self._encode_base_features(x, need_tokens=need_tokens)

        if self.adapter_type == "spatial_attention" and tokens is not None:
            logits = self.adapter(tokens, mask=mask)
        else:
            if mask is not None and tokens is not None and self.enable_masked_pooling:
                base_features = self._masked_pool(tokens, mask)
            logits = self.adapter(base_features)

        retrieval_features = F.normalize(logits, p=2, dim=1)

        if return_features:
            return retrieval_features
        if self.classifier is not None:
            return self.classifier(retrieval_features)
        return retrieval_features


class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int = 512, margin: float = 0.5, scale: float = 64):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.scale = scale
        self.ce = nn.CrossEntropyLoss()

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.ce(embeddings, labels)


def create_retrieval_model(
    clip_model_name: str = "ViT-B/32",
    num_classes: Optional[int] = None,
    embedding_dim: int = 512,
    device: str = "cuda",
    unfreeze_strategy: bool = True,
    adapter_type: str = "mlp",
    attention_hidden_dim: int = 256,
    attention_heads: int = 4,
    attention_dropout: float = 0.1,
    enable_masked_pooling: bool = True,
):
    device = device if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load(clip_model_name, device=device)
    model = RetrievalCLIPAdapter(
        clip_model=clip_model,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        unfreeze_strategy=unfreeze_strategy,
        adapter_type=adapter_type,
        attention_hidden_dim=attention_hidden_dim,
        attention_heads=attention_heads,
        attention_dropout=attention_dropout,
        enable_masked_pooling=enable_masked_pooling,
    ).to(device)
    return model


def _normalize_checkpoint(checkpoint: Dict) -> Dict:
    if "model_state_dict" in checkpoint:
        return checkpoint
    return {"model_state_dict": checkpoint}


def load_pretrained_model(model_path: str, device: str = "cuda"):
    device = device if torch.cuda.is_available() else "cpu"
    raw_checkpoint = torch.load(model_path, map_location=device)
    checkpoint = _normalize_checkpoint(raw_checkpoint)

    classes = checkpoint.get("classes", [])
    model = create_retrieval_model(
        clip_model_name=checkpoint.get("clip_model_name", "ViT-B/32"),
        num_classes=len(classes) if classes else checkpoint.get("num_classes"),
        embedding_dim=checkpoint.get("embedding_dim", 512),
        device=device,
        unfreeze_strategy=checkpoint.get("unfreeze_strategy", True),
        adapter_type=checkpoint.get("adapter_type", "mlp"),
        attention_hidden_dim=checkpoint.get("attention_hidden_dim", 256),
        attention_heads=checkpoint.get("attention_heads", 4),
        attention_dropout=checkpoint.get("attention_dropout", 0.1),
        enable_masked_pooling=checkpoint.get("enable_masked_pooling", True),
    )

    load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    metadata = {
        "best_accuracy": checkpoint.get("best_nn_accuracy", 0.0),
        "epoch": checkpoint.get("epoch", 0),
        "classes": classes,
        "unfreeze_strategy": checkpoint.get("unfreeze_strategy", True),
        "adapter_type": checkpoint.get("adapter_type", "mlp"),
        "enable_masked_pooling": checkpoint.get("enable_masked_pooling", True),
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
    }

    return model, classes, metadata
