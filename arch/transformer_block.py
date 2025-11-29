from arch.feed_forward import FeedForward
from arch.layer_norm import LayerNorm
from arch.attention import MultiHeadAttention
import torch.nn as nn
import torch
from arch.config import GPT_CONFIG_124M

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
           d_in=cfg["emb_dim"],
           d_out=cfg["emb_dim"],
           context_length=cfg["context_length"],
           num_heads=cfg["n_heads"],
           dropout=cfg["drop_rate"],
           qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.dropout(x)
        x = x + shortcut # Residual connection

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + shortcut # Residual connection

        return x

if __name__ == "__main__":
    torch.manual_seed(123)
    x = torch.rand(2, 4, 768)
    block = TransformerBlock(GPT_CONFIG_124M)
    output = block(x)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    