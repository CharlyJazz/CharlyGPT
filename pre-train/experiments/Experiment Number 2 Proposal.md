# Experiment Number 2 Proposal: Hacia un GPT más SOTA

## Resumen Ejecutivo

Este documento propone una **migración progresiva** de tu arquitectura GPT-2 actual hacia un diseño más moderno (estilo LLaMA / NanoGPT / Chinchilla). Los cambios están organizados en **4 fases** para que puedas validar cada mejora sin romper todo de golpe.

---

## Estado Actual de tu Arquitectura

### Archivos analizados
| Archivo | Descripción | Estado |
|---------|-------------|--------|
| `config.py` | Config 124M params | ✅ Bien estructurado |
| `gpt_model.py` | Modelo principal | ⚠️ Pos. embeddings absolutos |
| `transformer_block.py` | Bloque transformer | ✅ Pre-LN (correcto) |
| `attention.py` | Multi-head attention | ⚠️ Manual, sin SDPA |
| `feed_forward.py` | MLP clásico | ⚠️ GELU custom, 4x expansion |
| `layer_norm.py` | LayerNorm custom | ⚠️ Funcional pero no RMSNorm |
| `gelu.py` | GELU aproximado (tanh) | ⚠️ Crea tensores en forward |

### Diagnóstico detallado

#### 1. `gelu.py` - GELU custom
```python
# PROBLEMA: Crea tensor en cada forward (CPU por defecto)
torch.sqrt(torch.tensor(2.0 / torch.pi))
```
- **Riesgo**: device mismatch en GPU, overhead innecesario
- **Solución**: Usar `nn.GELU(approximate="tanh")` nativo

#### 2. `gpt_model.py` - Positional Embeddings
```python
self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
```
- **Limitación**: Embeddings absolutos aprendidos (estilo GPT-2 original)
- **SOTA actual**: RoPE (Rotary Position Embedding) - mejor generalización a longitudes no vistas

#### 3. `attention.py` - Attention manual
```python
attn_scores = queries @ keys.transpose(2, 3)
attn_scores.masked_fill_(mask_bool, -torch.inf)
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
```
- **Funciona** pero:
  - `-torch.inf` puede dar NaN en fp16
  - No aprovecha FlashAttention / SDPA de PyTorch 2.0+
  - Mask se recorta cada forward (menor issue)

#### 4. `layer_norm.py` - LayerNorm clásico
```python
mean = x.mean(dim=-1, keepdim=True)
var = x.var(dim=-1, keepdim=True, unbiased=False)
norm_x = (x - mean) / torch.sqrt(var + self.eps)
```
- **Funcional** pero modelos modernos usan **RMSNorm** (sin centrado de media)
- RMSNorm es ~10-15% más rápido y funciona igual o mejor

#### 5. `feed_forward.py` - MLP clásico
```python
nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
GELU(),
nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
```
- **Expansión 4x** es GPT-2 clásico
- **SOTA**: SwiGLU (3 matrices, mejor calidad por parámetro)

#### 6. `transformer_block.py` - Pre-LN ✅
```python
x = self.norm1(x)
x = self.att(x)
# ...residual
```
- **Correcto**: Ya usas Pre-LN (norma antes de attention/MLP)
- Esto es lo que usan los modelos modernos

#### 7. `gpt_model.py` - Weight Tying
```python
self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
```
- **No hay weight tying** entre `tok_emb` y `out_head`
- Activarlo reduce ~38M params y suele mejorar perplexity

---

## Propuesta: 4 Fases Progresivas

### FASE 1: Quick Wins (sin cambiar arquitectura)
**Objetivo**: Estabilidad + velocidad sin cambiar el comportamiento del modelo

| Cambio | Archivo | Impacto | Riesgo |
|--------|---------|---------|--------|
| GELU nativo | `feed_forward.py` | +velocidad, -bugs device | Ninguno |
| SDPA attention | `attention.py` | +velocidad, -memoria | Bajo |
| Weight tying | `gpt_model.py` | -38M params | Ninguno |

#### 1.1 GELU Nativo
```python
# feed_forward.py - ANTES
from arch.gelu import GELU
# ...
GELU(),

# feed_forward.py - DESPUÉS  
import torch.nn as nn
# ...
nn.GELU(approximate="tanh"),
```

#### 1.2 SDPA (Scaled Dot-Product Attention)
```python
# attention.py - ANTES
attn_scores = queries @ keys.transpose(2, 3)
mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
attn_scores.masked_fill_(mask_bool, -torch.inf)
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
attn_weights = self.dropout(attn_weights)
context_vec = (attn_weights @ values).transpose(1, 2)

# attention.py - DESPUÉS
import torch.nn.functional as F
# ...
context_vec = F.scaled_dot_product_attention(
    queries, keys, values,
    attn_mask=None,
    dropout_p=self.dropout.p if self.training else 0.0,
    is_causal=True
).transpose(1, 2)
```

#### 1.3 Weight Tying
```python
# gpt_model.py - en __init__, después de crear out_head
self.out_head.weight = self.tok_emb.weight
```

---

### FASE 2: RoPE (Rotary Position Embeddings)
**Objetivo**: Mejor generalización posicional, estándar en LLaMA/Mistral/etc.

#### Cambios requeridos

1. **Nuevo archivo**: `arch/rope.py`
```python
import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, seq_len):
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: (batch, heads, seq_len, head_dim)
    # cos, sin: (seq_len, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

2. **Modificar `attention.py`**: Aplicar RoPE a Q y K después de reshape

3. **Modificar `gpt_model.py`**: 
   - Remover/ignorar `self.pos_emb`
   - Pasar `rope` a los transformer blocks

---

### FASE 3: RMSNorm + SwiGLU
**Objetivo**: Mejoras de calidad típicas de LLaMA-style

#### 3.1 RMSNorm
```python
# arch/rmsnorm.py
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).type_as(x) * self.weight
```

#### 3.2 SwiGLU MLP
```python
# arch/feed_forward.py - versión SwiGLU
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUFeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden_dim = int(cfg["emb_dim"] * 8 / 3)  # ~2.67x para compensar 3 matrices
        hidden_dim = ((hidden_dim + 63) // 64) * 64  # múltiplo de 64 para eficiencia
        
        self.w1 = nn.Linear(cfg["emb_dim"], hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, cfg["emb_dim"], bias=False)
        self.w3 = nn.Linear(cfg["emb_dim"], hidden_dim, bias=False)
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

---

### FASE 4: Training Recipe Moderno
**Objetivo**: Hiperparámetros y técnicas de entrenamiento SOTA

#### 4.1 AdamW mejorado
```python
# Separar parámetros con/sin weight decay
decay_params = []
no_decay_params = []
for name, param in model.named_parameters():
    if param.requires_grad:
        if 'weight' in name and 'norm' not in name and 'emb' not in name:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

optimizer = torch.optim.AdamW([
    {'params': decay_params, 'weight_decay': 0.1},
    {'params': no_decay_params, 'weight_decay': 0.0}
], lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
```

#### 4.2 Mixed Precision (bf16 preferido)
```python
# Si tu GPU soporta bf16 (Ampere+)
scaler = None  # bf16 no necesita scaler
dtype = torch.bfloat16

# Forward con autocast
with torch.autocast(device_type='cuda', dtype=dtype):
    logits = model(input_ids)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
```

#### 4.3 Gradient checkpointing (opcional, ahorra VRAM)
```python
# En transformer_block.py
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    # Usar checkpoint para ahorrar memoria
    x = x + checkpoint(self._attn_block, x, use_reentrant=False)
    x = x + checkpoint(self._ff_block, x, use_reentrant=False)
    return x
```

---

## Comparativa: Tu Arquitectura vs SOTA

| Componente | Tu Actual | GPT-2 Original | LLaMA/Mistral | NanoGPT |
|------------|-----------|----------------|---------------|---------|
| Pos. Encoding | Absoluto (learned) | Absoluto (learned) | **RoPE** | Absoluto |
| Normalization | LayerNorm | LayerNorm | **RMSNorm** | LayerNorm |
| Norm Position | Pre-LN ✅ | Post-LN | Pre-LN | Pre-LN |
| Activation | GELU (tanh approx) | GELU | **SwiGLU** | GELU |
| MLP Expansion | 4x | 4x | **~2.67x + gate** | 4x |
| Attention | Manual | Manual | **SDPA/Flash** | SDPA |
| Weight Tying | No | Sí | No | Sí |
| QKV Bias | No | Sí | No | No |

---

## Plan de Implementación Sugerido

### Experimento 2A: Fase 1 solamente
- [ ] GELU nativo
- [ ] SDPA attention
- [ ] Weight tying
- **Validación**: Comparar val_loss y tokens/sec vs Experimento 1

### Experimento 2B: Fase 1 + Fase 2
- [ ] Todo lo anterior
- [ ] RoPE
- **Validación**: Mismas métricas + probar generalización a seq_len > 512

### Experimento 2C: Full SOTA (Fases 1-4)
- [ ] Todo lo anterior
- [ ] RMSNorm
- [ ] SwiGLU
- [ ] AdamW con betas=(0.9, 0.95)
- [ ] bf16 mixed precision
- **Validación**: Comparar final val_loss, tokens/sec, peak VRAM

---

## Criterios de Éxito

| Métrica | Baseline (Exp 1) | Target (Exp 2) |
|---------|------------------|----------------|
| Val Loss | (tu valor actual) | -5% o mejor |
| Tokens/sec | (tu valor actual) | +20% o mejor |
| Peak VRAM | (tu valor actual) | igual o menor |
| Estabilidad | ✅ | ✅ (sin NaN) |

---

## Checklist Final

### Fase 1 (Quick Wins)
- [ ] Reemplazar `from arch.gelu import GELU` → `nn.GELU(approximate="tanh")`
- [ ] Reemplazar attention manual → `F.scaled_dot_product_attention`
- [ ] Agregar weight tying: `self.out_head.weight = self.tok_emb.weight`
- [ ] Correr sanity check ~500 steps

### Fase 2 (RoPE)
- [ ] Crear `arch/rope.py`
- [ ] Modificar `attention.py` para aplicar RoPE
- [ ] Modificar `gpt_model.py` para remover pos_emb
- [ ] Correr sanity check ~500 steps

### Fase 3 (RMSNorm + SwiGLU)
- [ ] Crear `arch/rmsnorm.py`
- [ ] Modificar `feed_forward.py` para SwiGLU
- [ ] Reemplazar LayerNorm → RMSNorm en transformer_block y gpt_model
- [ ] Correr sanity check ~500 steps

### Fase 4 (Training Recipe)
- [ ] Separar param groups para weight decay
- [ ] Cambiar betas a (0.9, 0.95)
- [ ] Habilitar bf16/fp16 mixed precision
- [ ] Correr entrenamiento completo

---

---

## Análisis de NanoChat (Karpathy, Oct 2025)

### Características clave del modelo NanoChat
NanoChat es el proyecto más reciente de Karpathy para crear "el mejor ChatGPT que $100 pueden comprar".
Analicé el código fuente de `nanochat/gpt.py` y estas son las técnicas que usa:

| Componente | NanoChat | Tu Actual | Nota |
|------------|----------|-----------|------|
| **Positional Encoding** | RoPE (rotary) | Absoluto (learned) | ⚠️ Cambiar |
| **Normalization** | RMSNorm funcional (sin params!) | LayerNorm custom | ⚠️ Cambiar |
| **Norm Position** | Pre-LN + norm después de embedding | Pre-LN | ✅ Similar |
| **Activation (MLP)** | **ReLU² (squared)** | GELU tanh approx | 🆕 Interesante |
| **MLP Expansion** | 4x | 4x | ✅ Igual |
| **Attention** | SDPA con GQA | Manual | ⚠️ Cambiar |
| **Weight Tying** | **NO** (untied) | No | ✅ Igual |
| **QKV Bias** | No | No | ✅ Igual |
| **QK Norm** | **SÍ** (norm a Q y K) | No | 🆕 Nuevo |
| **Logit Softcap** | **SÍ** (15.0) | No | 🆕 Nuevo |
| **Optimizer** | **Muon + AdamW** (separados) | AdamW | 🆕 Avanzado |

### Código clave de NanoChat

#### 1. RMSNorm funcional (sin parámetros aprendibles)
```python
def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))
```
**Nota**: NanoChat usa `F.rms_norm` de PyTorch 2.4+ sin parámetros aprendibles (ni scale ni shift).

#### 2. RoPE (Rotary Position Embeddings)
```python
def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).to(x.dtype)
```

#### 3. QK Norm (normalizar Q y K antes de attention)
```python
q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
q, k = norm(q), norm(k)  # <-- QK Norm aquí
```
**Beneficio**: Estabiliza el entrenamiento, especialmente con secuencias largas.

#### 4. ReLU² en MLP (en lugar de GELU o SwiGLU)
```python
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()  # <-- ReLU² aquí
        x = self.c_proj(x)
        return x
```
**Nota**: ReLU² es más simple que SwiGLU y aparentemente funciona bien para modelos pequeños.

#### 5. SDPA con soporte GQA
```python
y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
```

#### 6. Logit Softcap (estabilidad numérica)
```python
softcap = 15
logits = self.lm_head(x)
logits = logits.float()
logits = softcap * torch.tanh(logits / softcap)  # <-- squash logits
```
**Beneficio**: Evita logits extremos que pueden causar inestabilidad.

#### 7. Norm después de token embedding
```python
x = self.transformer.wte(idx)
x = norm(x)  # <-- norm aquí, antes de los blocks
for block in self.transformer.h:
    x = block(x, cos_sin, kv_cache)
x = norm(x)
```

#### 8. Muon Optimizer (para matrices) + AdamW (para embeddings)
```python
# AdamW para embeddings y lm_head
adam_groups = [
    dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
    dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
]
adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)

# Muon para los linear layers del transformer
muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
muon_optimizer = Muon(matrix_params, **muon_kwargs)
```
**Nota**: Muon es un optimizador experimental de Karpathy. Para empezar, puedes usar solo AdamW.

#### 9. Inicialización especial
```python
# Zero init para c_proj y lm_head
torch.nn.init.zeros_(self.lm_head.weight)
torch.nn.init.zeros_(block.mlp.c_proj.weight)
torch.nn.init.zeros_(block.attn.c_proj.weight)
```

---

## Propuesta Actualizada: NanoChat-style

### FASE 1 (Quick Wins) - Sin cambios
- GELU nativo → `nn.GELU(approximate="tanh")`
- SDPA attention
- (opcional) Weight tying

### FASE 2 (RoPE) - Sin cambios
- Implementar RoPE
- Remover pos_emb absolutos

### FASE 2.5 (NanoChat additions) - **NUEVO**
- [ ] **QK Norm**: Agregar `norm(q), norm(k)` después de RoPE
- [ ] **Norm después de embedding**: `x = norm(self.tok_emb(idx))`
- [ ] **Logit softcap**: `logits = 15 * tanh(logits / 15)`

### FASE 3 (Norm + Activation) - **ACTUALIZADO**
Dos opciones:

**Opción A: LLaMA-style**
- RMSNorm con parámetros aprendibles
- SwiGLU MLP

**Opción B: NanoChat-style** (más simple)
- RMSNorm funcional (sin params): `F.rms_norm(x, (x.size(-1),))`
- ReLU² MLP: `F.relu(x).square()`

### FASE 4 (Training Recipe) - **ACTUALIZADO**
- AdamW con `betas=(0.8, 0.95)` (NanoChat usa 0.8, no 0.9)
- `eps=1e-10` (más pequeño que el default)
- Zero init para c_proj y lm_head
- (Avanzado) Muon optimizer para matrices

---

## Comparativa Final

| Componente | Tu Actual | LLaMA | NanoChat | Recomendación Exp 2 |
|------------|-----------|-------|----------|---------------------|
| Pos Encoding | Absoluto | RoPE | RoPE | **RoPE** |
| Norm | LayerNorm | RMSNorm (params) | RMSNorm (no params) | **RMSNorm** |
| Pre/Post LN | Pre-LN | Pre-LN | Pre-LN | Pre-LN ✅ |
| Activation | GELU | SwiGLU | ReLU² | **ReLU²** (simple) o SwiGLU |
| QK Norm | No | No | Sí | **Sí** |
| Logit Cap | No | No | Sí (15) | **Sí** |
| Attention | Manual | SDPA | SDPA+GQA | **SDPA** |
| Optimizer | AdamW | AdamW | Muon+AdamW | AdamW (0.8, 0.95) |

---

## Nueva Feature: Parametrización de GPT_CONFIG en YAML

### Problema Actual
La configuración del modelo está hardcodeada como constante en Python:

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,      # Vocabulary size
    "context_length": 1024,   # Context length
    "emb_dim": 768,           # Embedding dimension
    "n_heads": 12,            # Number of attention heads
    "n_layers": 12,           # Number of layers
    "drop_rate": 0.1,         # Dropout rate
    "qkv_bias": False         # Query-Key-Value bias
}
```

### Propuesta
Mover esta configuración al archivo YAML del experimento para permitir:
- Experimentar con diferentes tamaños de modelo sin modificar código
- Versionar configuraciones junto con los experimentos
- Facilitar reproducibilidad

### Ejemplo de YAML actualizado
```yaml
model:
  vocab_size: 50257
  context_length: 1024
  emb_dim: 768
  n_heads: 12
  n_layers: 12
  drop_rate: 0.1
  qkv_bias: false

training:
  # ... resto de config de training
```

---

## Feedback de Experto: Mariusz Kurman (@mkurman88)

### Comentario Original
> "Slightly shallow (stack more layers :P) and maybe too high dropout (I use 0.05 after 100B tokens with 0.0); weight decay?"

### Análisis del Feedback

| Aspecto | Tu Actual | Sugerencia | Nota |
|---------|-----------|------------|------|
| **Profundidad (n_layers)** | 12 | Más layers | Modelos más profundos generalizan mejor |
| **Dropout** | 0.1 | 0.05 → 0.0 | Reducir dropout conforme avanza el training |
| **Weight Decay** | ¿? | Revisar | Importante para regularización |

### Recomendaciones basadas en el feedback

#### 1. Dropout Schedule
- **Inicio**: 0.05 (no 0.1)
- **Después de ~100B tokens**: 0.0
- Esto permite más regularización al inicio y máxima capacidad al final

#### 2. Más Layers vs Más Width
- Con el mismo budget de parámetros, **más layers** suele ser mejor que **más width**
- Ejemplo: 16 layers con emb_dim=640 ≈ 12 layers con emb_dim=768

#### 3. Weight Decay
- Valor típico: **0.1** para AdamW
- Aplicar solo a weights de Linear layers (no a biases, norms, embeddings)
- Ya está en la Fase 4 del proposal:
```python
optimizer = torch.optim.AdamW([
    {'params': decay_params, 'weight_decay': 0.1},
    {'params': no_decay_params, 'weight_decay': 0.0}
], lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
```

### Configuración Sugerida para Experimento 2
```yaml
model:
  vocab_size: 50257
  context_length: 1024
  emb_dim: 768
  n_heads: 12
  n_layers: 16          # Aumentado de 12 → 16
  drop_rate: 0.05       # Reducido de 0.1 → 0.05
  qkv_bias: false

training:
  weight_decay: 0.1
  # Opcional: dropout schedule
  dropout_schedule:
    start: 0.05
    end: 0.0
    warmdown_tokens: 100_000_000_000  # 100B tokens
```

### Research https://convergentthinking.sh/posts/attention-normalizes-the-wrong-norm/

We treat attention as a solved primitive. You can fuse kernels, tile memory access, quantize weights, but the formula itself is finished.

It isn’t.

Softmax normalizes the L1 norm to 1. Variance preservation requires the L2 norm to equal 1. These constraints differ. The mismatch causes attention output variance to collapse as sequence length grows, forcing models to learn position-dependent compensation. That compensation doesn’t transfer to unseen lengths.

The fix is changing one norm.

With L1 softmax, output magnitude depends on both sparsity and sequence length. With L2 softmax, it doesn’t. The same holds for gradients.

---

## Referencias

- **RoPE**: [RoFormer paper](https://arxiv.org/abs/2104.09864)
- **SwiGLU**: [GLU Variants paper](https://arxiv.org/abs/2002.05202)
- **RMSNorm**: [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- **LLaMA**: [LLaMA paper](https://arxiv.org/abs/2302.13971)
- **NanoGPT**: [Karpathy's NanoGPT](https://github.com/karpathy/nanoGPT)
- **NanoChat**: [Karpathy's NanoChat](https://github.com/karpathy/nanochat) - Oct 2025
- [https://x.com/JordiNeil/status/2005035606536048980](https://x.com/JordiNeil/status/2005035606536048980)
- Remember to include "constraints" if the exercise is "rag". https://x.com/mkurman88/status/2004696093569987035

# ChatML

- Add new special tokens:

**Tokens oficiales de ChatML:**
- `<|im_start|>`
- `<|im_end|>`
- `<|think|>`
- `<|im_end|>`

![alt text](image.png)

Notas Extras:

1. "Slightly shallow (stack more layers)"
Tu configuración actual:

python
"n_layers": 12,  # En config.py
Recomendación:

Aumentar a 24-32 capas para el próximo experimento
neuroblast-v3 usa 80 capas y demostró que modelos profundos aprenden mejor de reasoning traces
Con SYNTH dataset (reasoning denso), más capas = mejor performance
2. "Maybe too high dropout (I use 0.05 after 100B tokens with 0.0)"
Tu configuración actual:

python
"drop_rate": 0.1,  # 10% dropout en config.py
Recomendación del pro player:

Fase 1 (0-100B tokens): drop_rate = 0.0 (sin dropout)
Fase 2 (100B+ tokens): drop_rate = 0.05 (5% dropout)
Razón: Dropout alto al inicio dificulta el aprendizaje. Mejor dejar que el modelo aprenda primero, luego agregar regularización.

Cambio en el map de synth:

Haha "Remember to include "constraints" if the exercise is "rag"." i got that tip


def map_synth(x):
    return {
        "text": "<|im_start|>user\n"
        + x["query"] + (f"\n{x['constraints']}" if 'rag' in x['exercise'] else '')
        + "<|im_end|>\n"
        + "<|im_start|>

research\neuroblast-v3\train\train.py