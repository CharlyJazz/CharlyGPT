
import torch.nn as nn
import torch
from arch.transformer_block import TransformerBlock
from arch.layer_norm import LayerNorm
from arch.rope import RotaryEmbedding, apply_rotary_pos_emb

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.use_rope = cfg.get("use_rope", False)
        if self.use_rope:
            print("Using RoPE")
            # Crear RoPE (sin parámetros entrenables)
            head_dim = cfg["emb_dim"] // cfg["n_heads"]
            self.rope = RotaryEmbedding(
                dim=head_dim,
                max_seq_len=cfg["context_length"],
                base=cfg.get("rope_base", 10000)
            )
        else:
            self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(
                cfg=cfg,
                rope=self.rope if self.use_rope else None
            ) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )   

        if cfg.get("use_weight_tying", False):
            self.out_head.weight = self.tok_emb.weight
    
    def forward(self, in_idx: torch.Tensor):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        if self.use_rope:
            x = tok_embeds
        else:
            pos_embeds = self.pos_emb(
                torch.arange(seq_len, device=in_idx.device)
            )
            x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

if __name__ == "__main__":
    cfg = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }
    model = GPTModel(cfg)
    in_idx = torch.randint(0, cfg["vocab_size"], (2, 4))
    logits = model(in_idx)
    print("Input shape:", in_idx.shape)
    print("Logits shape:", logits.shape)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")
    print("Token embedding layer shape:", model.tok_emb.weight.shape)
    print("Output layer shape:", model.out_head.weight.shape)
    total_params_gpt2 = (
        total_params - sum(
            p.numel()
            for p in model.out_head.parameters()
        )
    )
    print(f"Number of trainable parameters "
    f"considering weight tying: {total_params_gpt2:,}")

    total_size_bytes = total_params * 4
    total_size_mb = total_size_bytes / (1024 * 1024)
    print(f"Total size of the model: {total_size_mb:.2f} MB")