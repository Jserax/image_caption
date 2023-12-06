from typing import Union

import torch
from torch import nn
from torch.nn import functional as F


class TextPositionalEncoding(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        max_len: int,
    ):
        super().__init__()
        position = torch.arange(max_len).view(-1, 1)
        div_term = torch.exp(
            torch.arange(0, emb_dim, 2) * (-torch.log(torch.tensor(max_len)) / emb_dim)
        )
        pos = torch.zeros(max_len, emb_dim)
        pos[:, 0::2] = torch.sin(position * div_term)
        pos[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos", pos)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = x + self.pos[None, : x.size(1)]
        return x


class CrossMultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_heads: int,
        emb_dim: int,
        dropout: float,
    ) -> None:
        super(CrossMultiHeadAttention, self).__init__()
        if emb_dim % n_heads != 0:
            raise Exception(
                f"Embedding dim(emb_dim): {emb_dim} must be divisible by number of heads(n_heads): {n_heads}"
            )
        self.n_heads = n_heads
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // n_heads
        self.dropout = dropout
        self.to_q = nn.Linear(emb_dim, emb_dim)
        self.to_kv = nn.Linear(emb_dim, 2 * emb_dim)
        self.out = nn.Linear(emb_dim, emb_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc: torch.Tensor,
        attn: Union[None, torch.FloatTensor] = None,
    ) -> torch.Tensor:
        B, S, D = x.size()
        q = self.to_q(x)
        k, v = self.to_kv(enc).chunk(2, dim=-1)
        q = q.view(B, S, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn,
            dropout_p=self.dropout if self.training else 0,
            is_causal=False,
        )
        out = out.permute(0, 2, 1, 3).reshape(B, S, D)
        out = self.out_dropout(out)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_heads: int,
        emb_dim: int,
        dropout: float,
        causal: bool,
    ) -> None:
        super(MultiHeadAttention, self).__init__()
        if emb_dim % n_heads != 0:
            raise Exception(
                f"Embedding dim(emb_dim): {emb_dim} must be divisible by number of heads(n_heads): {n_heads}"
            )
        self.n_heads = n_heads
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // n_heads
        self.dropout = dropout
        self.causal = causal
        self.to_qkv = nn.Linear(emb_dim, 3 * emb_dim)
        self.out = nn.Linear(emb_dim, emb_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, attn: Union[None, torch.FloatTensor] = None
    ) -> torch.Tensor:
        B, S, D = x.size()
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = q.view(B, S, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, S, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, S, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn,
            dropout_p=self.dropout if self.training else 0,
            is_causal=self.causal,
        )
        out = out.permute(0, 2, 1, 3).reshape(B, S, D)
        out = self.out_dropout(out)
        return out


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        emb_dim: int,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.conv = nn.Conv2d(
            3,
            emb_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.pos_emb = nn.Parameter(torch.randn(1, self.num_patches, emb_dim))

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # (batch_size, in_ch, h, w) -> (batch_size, num_patches, emb_dim)
        B = x.size(0)
        x = self.conv(x).view(B, self.num_patches, self.emb_dim)
        x = x + self.pos_emb
        return x
