import torch
import torch.nn as nn

# This specific implementation of layer normalization operates on the last dimension of
# the input tensor x, which represents the embedding dimension (emb_dim). The variable eps is a small constant (epsilon) added to the variance to prevent division by zero
# during normalization. The scale and shift are two trainable parameters (of the
# same dimension as the input) that the LLM automatically adjusts during training if it
# is determined that doing so would improve the model’s performance on its training
# task. This allows the model to learn appropriate scaling and shifting that best suit the
# data it is processing
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(
            torch.ones(emb_dim)
        )
        self.shift = nn.Parameter(
            torch.zeros(emb_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift



if __name__ == "__main__":
    torch.manual_seed(123)
    torch.set_printoptions(sci_mode=False)
    batch_example = torch.rand(2, 5)
    ln = LayerNorm(emb_dim=5)
    out_ln = ln(batch_example)
    mean = out_ln.mean(dim=-1, keepdim=True)
    var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
    print("Mean:\n", mean)
    print("Variance:\n", var)