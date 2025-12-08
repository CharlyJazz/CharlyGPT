"""
Tokenizer Utilities for GPT Pre-training

Provides a consistent interface to the tiktoken GPT-2 tokenizer.
"""

import torch
import tiktoken
from typing import Optional, List, Union
from functools import lru_cache


@lru_cache(maxsize=1)
def get_tokenizer(encoding_name: str = "gpt2") -> tiktoken.Encoding:
    """
    Get the tiktoken tokenizer (cached).
    
    Args:
        encoding_name: Name of the encoding to use (default: "gpt2")
    
    Returns:
        tiktoken.Encoding instance
    """
    return tiktoken.get_encoding(encoding_name)


def get_eos_token_id(tokenizer: Optional[tiktoken.Encoding] = None) -> int:
    """
    Get the end-of-sequence token ID.
    
    Args:
        tokenizer: Optional tokenizer instance. If None, uses default GPT-2 tokenizer.
    
    Returns:
        EOS token ID (50256 for GPT-2)
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()
    
    eos_tokens = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})
    return eos_tokens[0]


def text_to_token_ids(
    text: str, 
    tokenizer: Optional[tiktoken.Encoding] = None,
    add_batch_dim: bool = True
) -> torch.Tensor:
    """
    Convert text string to token IDs tensor.
    
    Args:
        text: Input text string
        tokenizer: Optional tokenizer instance. If None, uses default GPT-2 tokenizer.
        add_batch_dim: If True, adds batch dimension (shape: [1, seq_len])
    
    Returns:
        Tensor of token IDs
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()
    
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    tensor = torch.tensor(encoded, dtype=torch.long)
    
    if add_batch_dim:
        tensor = tensor.unsqueeze(0)
    
    return tensor


def token_ids_to_text(
    token_ids: Union[torch.Tensor, List[int]], 
    tokenizer: Optional[tiktoken.Encoding] = None
) -> str:
    """
    Convert token IDs back to text string.
    
    Args:
        token_ids: Tensor or list of token IDs
        tokenizer: Optional tokenizer instance. If None, uses default GPT-2 tokenizer.
    
    Returns:
        Decoded text string
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()
    
    if isinstance(token_ids, torch.Tensor):
        if token_ids.dim() > 1:
            token_ids = token_ids.squeeze()
        token_ids = token_ids.tolist()
    
    return tokenizer.decode(token_ids)


def get_vocab_size(tokenizer: Optional[tiktoken.Encoding] = None) -> int:
    """
    Get the vocabulary size of the tokenizer.
    
    Args:
        tokenizer: Optional tokenizer instance
    
    Returns:
        Vocabulary size (50257 for GPT-2)
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()
    
    return tokenizer.n_vocab


def count_tokens(text: str, tokenizer: Optional[tiktoken.Encoding] = None) -> int:
    """
    Count the number of tokens in a text string.
    
    Args:
        text: Input text string
        tokenizer: Optional tokenizer instance
    
    Returns:
        Number of tokens
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()
    
    return len(tokenizer.encode(text, allowed_special={"<|endoftext|>"}))
