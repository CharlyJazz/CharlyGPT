import torch
import tiktoken
from typing import Optional
from torch import nn
from torch.utils.data import Dataset, DataLoader

tokenizer: tiktoken.Encoding = tiktoken.get_encoding("gpt2")

def text_to_token_ids(text: str, tokenizer: tiktoken.Encoding) -> torch.Tensor:
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids: torch.Tensor, tokenizer: tiktoken.Encoding) -> str:
    """Convert token IDs tensor to text"""
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def generate_text_simple(model: nn.Module, idx: torch.Tensor, max_new_tokens: int, context_size: int,
             temperature: float = 0.0, top_k: Optional[int] = None, eos_id: Optional[int] = None):
    """Generate text using the given model and input"""
    # Get model device
    device = next(model.parameters()).device
    idx = idx.to(device)
    
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
        
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        
        # Ensure idx_next is on the same device as idx
        idx_next = idx_next.to(idx.device)
        
        if idx_next == eos_id:
            break
        
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx