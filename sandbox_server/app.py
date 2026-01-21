from __future__ import annotations

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, Optional

import torch
import tiktoken
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from arch.gpt_model import GPTModel

LOG_FILE = Path(os.getenv("CHAT_LOG_FILE", "sandbox_server/chat.log"))
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(message)s",
    encoding="utf-8",
)
chat_logger = logging.getLogger("chat")

app = FastAPI(title="Model Sandbox API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"] ,
    allow_headers=["*"],
)

EXPERIMENT_PATH = Path(
    os.getenv(
        "EXPERIMENT_YAML",
        "pre-train/experiments/Experiment2-ChatML-Optimizations.yaml",
    )
)


def load_experiment_cfg(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_checkpoint_path(cfg: Dict[str, object]) -> Path:
    override = os.getenv("CHECKPOINT_PATH")
    if override:
        return Path(override)
    storage = cfg.get("storage", {})
    base_folder = Path(storage.get("base_folder", "."))
    experiment_name = cfg.get("experiment_name", "Experiment")
    checkpoint_name = storage.get("checkpoint_to_resume")
    if not checkpoint_name:
        raise ValueError("checkpoint_to_resume missing in experiment yaml")
    return base_folder / experiment_name / "checkpoints" / checkpoint_name


tokenizer = tiktoken.get_encoding("gpt2")


def text_to_token_ids(text: str) -> torch.Tensor:
    encoded = tokenizer.encode(text, allowed_special={"<|END_OF_TEXT|>"})
    return torch.tensor(encoded).unsqueeze(0)


def token_ids_to_text(token_ids: torch.Tensor) -> str:
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


class InferenceWrapper:
    def __init__(self, cfg: Dict[str, object]):
        self.model_cfg = dict(cfg["model"])
        device_name = cfg.get("hardware", {}).get("device", "cpu")
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requerido, pero torch.cuda.is_available() es False")
        self.device = torch.device(device_name)
        checkpoint_path = resolve_checkpoint_path(cfg)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if "pos_emb.weight" in state_dict:
            self.model_cfg["use_rope"] = False
        self.model = GPTModel(self.model_cfg).to(self.device)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if unexpected:
            print(f"[WARN] Unexpected keys in checkpoint: {unexpected}")
        if missing:
            print(f"[WARN] Missing keys in checkpoint: {missing}")
        self.model.eval()
        self.context_length = self.model_cfg.get("context_length", 1024)

    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float = 0.7,
        top_k: Optional[int] = 50,
        stop_tokens: Optional[list] = None,
    ) -> Generator[str, None, None]:
        if stop_tokens is None:
            stop_tokens = ["<|im_end|>", "<|endoftext|>"]
        idx = text_to_token_ids(prompt).to(self.device)
        generated_text = ""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_length :]
            with torch.no_grad():
                logits = self.model(idx_cond)
            logits = logits[:, -1, :]
            if top_k is not None:
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1].unsqueeze(-1)
                logits = torch.where(
                    logits < min_val,
                    torch.tensor(float("-inf"), device=logits.device),
                    logits,
                )
            if temperature > 0.0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            idx_next = idx_next.to(idx.device)
            idx = torch.cat((idx, idx_next), dim=1)
            token_text = token_ids_to_text(idx_next)
            generated_text += token_text
            if any(stop in generated_text for stop in stop_tokens):
                break
            yield token_text


experiment_cfg = load_experiment_cfg(EXPERIMENT_PATH)
inference = InferenceWrapper(experiment_cfg)


class ChatRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_k: Optional[int] = None


def format_chatml_prompt(user_message: str) -> str:
    return f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"


def stream_generate_with_log(payload: ChatRequest) -> Generator[str, None, None]:
    timestamp = datetime.now().isoformat()
    chat_logger.info(f"\n{'='*60}")
    chat_logger.info(f"[{timestamp}] USER:")
    chat_logger.info(payload.prompt)
    chat_logger.info(f"\n[{timestamp}] ASSISTANT:")

    formatted_prompt = format_chatml_prompt(payload.prompt)
    full_response = []
    for chunk in inference.stream_generate(
        formatted_prompt,
        payload.max_new_tokens,
        temperature=payload.temperature,
        top_k=payload.top_k,
    ):
        full_response.append(chunk)
        yield chunk

    chat_logger.info("".join(full_response))
    chat_logger.info(f"{'='*60}\n")


@app.post("/chat")
async def chat(payload: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        stream_generate_with_log(payload),
        media_type="text/plain",
    )
