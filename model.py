# model.py - LightGastroFormer architecture

import torch
import torch.nn as nn
import torch.nn.functional as F


class EfficientAttention(nn.Module):
    """Standard multi-head self-attention with dropout."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        self.num_heads = num_heads
        self.scale     = (dim // num_heads) ** -0.5

        self.qkv       = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)                          # B, H, N, head_dim

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MedicalGatedFeedForward(nn.Module):
    """Gated FFN tailored for medical imaging feature spaces."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.gate = nn.Linear(in_features, hidden_features)
        self.fc2  = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x) * torch.sigmoid(self.gate(x))
        x = F.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LightTransformerBlock(nn.Module):
    """Single transformer block: LayerNorm → Attention + LayerNorm → Gated FFN."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        qkv_bias: bool   = False,
        drop: float      = 0.0,
        attn_drop: float = 0.0,
        mlp_drop: float  = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = EfficientAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden     = int(dim * mlp_ratio)
        self.mlp   = MedicalGatedFeedForward(dim, hidden, dim, drop=mlp_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MultiResPatchwiseTokenizer(nn.Module):
    """
    Dual-resolution patch tokenizer.

    Combines coarse (patch_size) and fine (patch_size // 2) patch embeddings
    to preserve both global context and local GI texture details.
    """

    def __init__(
        self,
        img_size: int   = 224,
        patch_size: int = 16,
        in_chans: int   = 3,
        embed_dim: int  = 384,
    ) -> None:
        super().__init__()
        self.img_size   = img_size
        self.patch_size = patch_size
        self.grid_size  = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.embed_dim  = embed_dim

        # Coarse (primary) embedding
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim // 2,
            kernel_size=patch_size, stride=patch_size,
        )
        # Fine (secondary) embedding
        self.small_patch_embed = nn.Conv2d(
            in_chans, embed_dim // 4,
            kernel_size=patch_size // 2, stride=patch_size // 2,
        )
        self.small_patch_pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fusion   = nn.Linear(embed_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.norm      = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, (
            f"Expected {self.img_size}×{self.img_size}, got {H}×{W}."
        )

        # Coarse tokens: B, N, embed_dim//2
        primary = self.patch_embed(x).flatten(2).transpose(1, 2)

        # Fine tokens: B, N, embed_dim//4  (pooled down to same grid)
        secondary = self.small_patch_pooling(
            self.small_patch_embed(x)
        ).flatten(2).transpose(1, 2)

        # Trim if fine grid is slightly larger
        if secondary.size(1) > primary.size(1):
            secondary = secondary[:, :primary.size(1), :]

        # Pad fine tokens to embed_dim//2 with zeros, then concat → embed_dim
        pad = torch.zeros(B, secondary.size(1), self.embed_dim // 4, device=x.device)
        secondary = torch.cat([secondary, pad], dim=2)
        tokens = self.fusion(torch.cat([primary, secondary], dim=2))

        # Prepend [CLS] and add positional embeddings
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, tokens], dim=1)
        x   = x + self.pos_embed[:, :x.size(1), :]
        return self.norm(x)


class LightGastroFormer(nn.Module):
    """
    Lightweight Vision Transformer for gastrointestinal disease classification.

    Uses a dual-resolution tokenizer, shallow transformer blocks, and an
    auxiliary classification head on averaged patch tokens for regularisation.
    """

    def __init__(
        self,
        img_size: int    = 224,
        patch_size: int  = 8,
        in_chans: int    = 3,
        num_classes: int = 8,
        embed_dim: int   = 384,
        depth: int       = 8,
        num_heads: int   = 6,
        mlp_ratio: float = 2.0,
        qkv_bias: bool   = True,
        drop_rate: float      = 0.1,
        attn_drop_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes  = num_classes
        self.num_features = embed_dim
        self.has_logits   = True

        self.patch_embed = MultiResPatchwiseTokenizer(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim,
        )
        self.blocks = nn.Sequential(*[
            LightTransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, mlp_drop=drop_rate,
            )
            for _ in range(depth)
        ])
        self.norm     = nn.LayerNorm(embed_dim)
        self.head     = nn.Linear(embed_dim, num_classes)
        self.aux_head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)

    def forward(self, x: torch.Tensor):
        x        = self.patch_embed(x)
        x        = self.blocks(x)
        x        = self.norm(x)

        cls_out  = self.head(x[:, 0])                   # [CLS] token
        aux_out  = self.aux_head(x[:, 1:].mean(dim=1))  # mean of patch tokens
        return cls_out, aux_out