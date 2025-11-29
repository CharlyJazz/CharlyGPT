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

class GPTDatasetV1(Dataset):
    def __init__(self, txt: str, tokenizer: tiktoken.Encoding, max_length: int, stride: int):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={'<|endoftext|>'})
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(
    txt: str, 
    batch_size: int = 4, 
    max_length: int = 256,
    stride: int = 128, 
    shuffle: bool = True, 
    drop_last: bool = True,
    num_workers: int = 0
) -> DataLoader:
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    assert len(dataloader) > 0
    assert isinstance(dataloader, DataLoader)
    return dataloader