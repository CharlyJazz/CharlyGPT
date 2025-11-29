import torch
import torch.nn as nn
from arch.gelu import GELU

# FeedForward module is a small neural network consisting of two
# Linear layers and a GELU activation function. In the 124-million-parameter GPT
# model, it receives the input batches with tokens that have an embedding size of 768
# each via the GPT_CONFIG_124M dictionary where GPT_CONFIG_ 124M["emb_dim"] = 768.
class FeedForward(nn.Module):
    """The FeedForward module plays a crucial role in enhancing the model’s ability to learn
    from and generalize the data. Although the input and output dimensions of this
    module are the same, it internally expands the embedding dimension into a higherdimensional 
    space through the first linear layer. 
    This expansion is followed by a nonlinear GELU activation and then a contraction back to 
    the original dimension with the second linear transformation. 
    Such a design allows for the exploration of a richer representation space."""
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


if __name__ == "__main__":
    torch.manual_seed(123)
    torch.set_printoptions(sci_mode=False)
    ffn = FeedForward(cfg={"emb_dim": 768})
    x = torch.randn(2, 3, 768)
    out = ffn(x)
    print(out.shape)
