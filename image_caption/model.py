import torch
from torch import nn
from torch.nn import functional as F

from layers import (
    CrossMultiHeadAttention,
    MultiHeadAttention,
    PatchEmbedding,
    TextPositionalEncoding,
)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        n_heads: int,
        emb_dim: int,
        ff_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(emb_dim)
        self.layernorm2 = nn.LayerNorm(emb_dim)
        self.attn = MultiHeadAttention(
            n_heads,
            emb_dim,
            dropout,
            False,
        )
        self.ff = nn.Sequential(
            nn.Linear(emb_dim, ff_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ff_dim, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.layernorm1(x))
        x = x + self.ff(self.layernorm2(x))
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        n_layers: int,
        n_heads: int,
        emb_dim: int,
        ff_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.transformers = nn.ModuleList(
            [
                EncoderLayer(
                    n_heads,
                    emb_dim,
                    ff_dim,
                    dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.patch_embedding = PatchEmbedding(
            img_size,
            patch_size,
            emb_dim,
        )
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(emb_dim)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.patch_embedding(x)
        x = self.dropout(x)
        res = x
        for block in self.transformers:
            x = block(x)
        x = x + res
        out = self.layernorm(x)
        return out


class DecoderLayer(nn.Module):
    def __init__(
        self,
        n_heads: int,
        emb_dim: int,
        ff_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(emb_dim)
        self.layernorm2 = nn.LayerNorm(emb_dim)
        self.layernorm3 = nn.LayerNorm(emb_dim)
        self.attn = MultiHeadAttention(
            n_heads,
            emb_dim,
            dropout,
            True,
        )

        self.cross_attn = CrossMultiHeadAttention(
            n_heads,
            emb_dim,
            dropout,
        )
        self.ff = nn.Sequential(
            nn.Linear(emb_dim, ff_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ff_dim, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.FloatTensor, enc: torch.FloatTensor
    ) -> torch.FloatTensor:
        x = self.layernorm1(x + self.attn(x))
        x = self.layernorm2(x + self.cross_attn(x, enc, None))
        x = self.layernorm3(x + self.ff(x))
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        max_len: int,
        vocab_size: int,
        n_layers: int,
        n_heads: int,
        emb_dim: int,
        ff_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.transformers = nn.ModuleList(
            [
                DecoderLayer(
                    n_heads,
                    emb_dim,
                    ff_dim,
                    dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.word_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = TextPositionalEncoding(emb_dim, max_len)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, vocab_size)

    def forward(self, x: torch.LongTensor, enc: torch.FloatTensor) -> torch.FloatTensor:
        x = self.word_emb(x)
        x = self.pos_emb(x)
        x = self.dropout(x)
        for block in self.transformers:
            x = block(x, enc)
        x = self.layernorm(x)
        out = self.head(x)
        return out


class CaptionModel(nn.Module):
    def __init__(
        self,
        img_size: int = 128,
        patch_size: int = 8,
        max_len: int = 128,
        vocab_size: int = 16384,
        enc_layers: int = 4,
        enc_heads: int = 8,
        enc_dim: int = 256,
        enc_ff_dim: int = 512,
        dec_layers: int = 6,
        dec_heads: int = 8,
        dec_dim: int = 256,
        dec_ff_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            img_size,
            patch_size,
            enc_layers,
            enc_heads,
            enc_dim,
            enc_ff_dim,
            dropout,
        )
        self.decoder = Decoder(
            max_len,
            vocab_size,
            dec_layers,
            dec_heads,
            dec_dim,
            dec_ff_dim,
            dropout,
        )

    def forward(
        self, img: torch.FloatTensor, text: torch.LongTensor
    ) -> torch.FloatTensor:
        enc = self.encoder(img)
        out = self.decoder(text, enc)
        return out

    @torch.no_grad()
    def generate(
        self, img: torch.FloatTensor, max_len: int, temp: float = 1.0
    ) -> torch.LongTensor:
        idx = torch.tensor([101], device=self.device)  # cls token for BOS
        for _ in range(max_len):
            logits = self(img, idx)
            logits = logits[:, -1, :] / temp
            probs = F.softmax(logits, dim=-1)
            next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next], dim=-1)
        return idx

    def num_params(self) -> int:
        return sum([p.numel() for p in self.parameters() if p.requires_grad])
