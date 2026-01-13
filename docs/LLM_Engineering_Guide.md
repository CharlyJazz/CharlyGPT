# LLM Engineering Guide for Interviews

Una guía completa sobre conceptos clave de LLM engineering, cubriendo arquitectura, entrenamiento, y mejores prácticas modernas.

---

## Tabla de Contenidos

1. [Diferencias: LLMs vs Computer Vision](#1-diferencias-llms-vs-computer-vision)
2. [Scaling Laws y Chinchilla](#2-scaling-laws-y-chinchilla)
3. [Learning Rate Schedules](#3-learning-rate-schedules)
4. [Arquitectura Moderna de Transformers](#4-arquitectura-moderna-de-transformers)
5. [Optimizers para LLMs](#5-optimizers-para-llms)
6. [Data y Tokenización](#6-data-y-tokenización)
7. [Training Recipes](#7-training-recipes)
8. [Inference y Serving](#8-inference-y-serving)
9. [Preguntas Comunes de Entrevista](#9-preguntas-comunes-de-entrevista)

---

## 1. Diferencias: LLMs vs Computer Vision

### ¿Por qué el entrenamiento de LLMs es diferente?

| Aspecto | Computer Vision (OCR, etc.) | LLMs |
|---------|----------------------------|------|
| **Dataset típico** | 10K - 1M imágenes | 1T+ tokens |
| **Epochs comunes** | 50-200+ epochs | < 1 - 4 epochs |
| **Data augmentation** | Muy efectiva (rotación, flip, crop) | Casi imposible |
| **Overfitting** | Controlado con augmentation | Alto riesgo de memorización |
| **Objetivo** | Clasificar/detectar patrones | Modelar distribución de lenguaje |

### ¿Por qué menos epochs en LLMs?

1. **No hay augmentation efectiva para texto**
   - Ver "The cat sat on the mat" 100 veces = memorizar esa frase
   - En vision: ver una imagen 100 veces con augmentation ≈ 100 imágenes diferentes

2. **El objetivo es diferente**
   - Vision: Aprender features invariantes
   - LLM: Predecir el siguiente token exacto (sensible a cada carácter)

3. **Datos únicos > datos repetidos**
   - Chinchilla demostró que es más eficiente ver más datos únicos
   - Repetir datos lleva a memorización, no generalización

### Data Augmentation en NLP (limitada)

| Técnica | Ejemplo | Problema |
|---------|---------|----------|
| Typos/errores | "hello" → "helo" | El modelo aprende que errores son válidos |
| Sinónimos | "happy" → "joyful" | Cambia significado sutilmente |
| Back-translation | EN→ES→EN | Costoso, pierde matices |
| Word dropout | "the cat sat" → "the _ sat" | Usado en BERT, no en GPT |

**Lo que SÍ funciona:**
- Más datos (web scraping, libros, código)
- Datos sintéticos (generar con otro LLM)
- Curriculum learning (fácil → difícil)

---

## 2. Scaling Laws y Chinchilla

### Chinchilla Scaling Law (DeepMind, 2022)

La regla de Chinchilla establece que para un budget de compute óptimo:

```
tokens_óptimos ≈ 20 × num_parameters
```

**Ejemplo:**
- Modelo de 163M params → entrenar con ~3.26B tokens
- Modelo de 7B params → entrenar con ~140B tokens

### Implicaciones prácticas

| Antes de Chinchilla | Después de Chinchilla |
|---------------------|----------------------|
| Modelos más grandes = mejor | Balance tamaño vs datos |
| GPT-3 175B con 300B tokens | LLaMA 7B con 1T tokens |
| Undertrained pero grande | Fully trained pero eficiente |

### Compute-Optimal Training

```python
# Fórmula aproximada
C = 6 * N * D  # FLOPs totales

# Donde:
# C = Compute (FLOPs)
# N = Número de parámetros
# D = Número de tokens de entrenamiento
```

Para training óptimo: `N ≈ D / 20`

---

## 3. Learning Rate Schedules

### Schedule Tradicional: Cosine Decay

```
LR
│  ╭──╮
│ ╱    ╲
│╱      ╲____
└──────────────► steps
 warmup   cosine decay continuo
```

**Fórmula:**
```python
if step < warmup_steps:
    lr = max_lr * step / warmup_steps
else:
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * progress))
```

### Schedule NanoChat/Moderno: Warmup → Constant → Warmdown

```
LR
│     ┌────────────────┐
│    ╱                  ╲
│   ╱                    ╲___
└───────────────────────────► steps
 5%        75%           20%
warmup   CONSTANT      warmdown
```

**Fórmula:**
```python
def get_lr_for_step(step, total_steps, min_lr, max_lr):
    warmup_iters = 0.05 * total_steps      # primeros 5%
    warmdown_iters = 0.20 * total_steps    # últimos 20%
    final_lr_frac = min_lr / max_lr
    
    if step < warmup_iters:
        return max_lr * (step + 1) / warmup_iters
    elif step <= total_steps - warmdown_iters:
        return max_lr  # Constant
    else:
        progress = (total_steps - step) / warmdown_iters
        return max_lr * (progress + (1 - progress) * final_lr_frac)
```

### ¿Cuál es mejor?

| Cosine Decay | NanoChat Warmdown |
|--------------|-------------------|
| Más suave | LR alto más tiempo |
| Estándar en papers | Más agresivo al final |
| Empieza a decaer temprano | Mantiene capacidad de aprendizaje |

### Stateful vs Stateless Schedulers

| Stateful (PyTorch default) | Stateless (NanoChat style) |
|---------------------------|---------------------------|
| `scheduler.step()` modifica estado | Calcula LR directo del step |
| Resume puede fallar | Resume siempre funciona |
| Depende de `state_dict` | Solo necesita `global_step` |

**Recomendación:** Usar stateless para robustez en resume.

---

## 4. Arquitectura Moderna de Transformers

### Comparativa: GPT-2 vs LLaMA vs NanoChat

| Componente | GPT-2 | LLaMA | NanoChat |
|------------|-------|-------|----------|
| Pos. Encoding | Absoluto (learned) | RoPE | RoPE |
| Normalization | LayerNorm | RMSNorm (params) | RMSNorm (no params) |
| Norm Position | Post-LN | Pre-LN | Pre-LN |
| Activation | GELU | SwiGLU | ReLU² |
| MLP Expansion | 4x | ~2.67x + gate | 4x |
| Attention | Manual | SDPA | SDPA + GQA |
| QK Norm | No | No | Sí |
| Weight Tying | Sí | No | No |

### RoPE (Rotary Position Embeddings)

#### ¿Qué es y para qué sirve?

**Respuesta para entrevista (30 segundos):**
> "RoPE codifica posiciones mediante rotaciones en el espacio de embeddings. Rota queries y keys proporcionalmente a su posición, de modo que el producto punto Q·K depende de la distancia relativa entre tokens. Esto permite que el modelo generalice mejor a secuencias largas sin parámetros adicionales. Es la técnica estándar en Llama y Mistral."

#### ¿Por qué solo Q y K, no V?

- **Q y K con RoPE**: Para calcular "quién debe atender a quién" considerando posiciones relativas
- **V sin RoPE**: Para proporcionar contenido puro sin bias posicional
- La información posicional ya fue usada en Q y K para calcular los attention weights
- Aplicar RoPE a V introduciría bias posicional innecesario en el contenido

#### Relación con rotaciones 2D clásicas

RoPE es exactamente la matriz de rotación 2D aplicada eficientemente:

**Matriz de rotación clásica:**
```
[x']   [cos(θ)  -sin(θ)]   [x]
[y'] = [sin(θ)   cos(θ)] × [y]

Expandiendo:
x' = x * cos(θ) - y * sin(θ)
y' = x * sin(θ) + y * cos(θ)
```

**RoPE hace exactamente esto:**
```python
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)  # [x, y]
    return torch.cat((-x2, x1), dim=-1)  # [-y, x]

q_embed = (q * cos) + (rotate_half(q) * sin)
# Resultado: [q₀*cos(θ) - q₁*sin(θ), q₁*cos(θ) + q₀*sin(θ)]
```

Para `head_dim=64`, RoPE aplica **32 rotaciones 2D independientes**, una por cada par de dimensiones.

#### Implementación completa

```python
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        # Frecuencias inversas: dimensiones bajas rotan rápido, altas rotan lento
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len):
        # Pre-computar cos/sin para eficiencia
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim//2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, dim)
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

#### Integración con SDPA

RoPE es completamente compatible con `F.scaled_dot_product_attention`:

```python
# 1. Proyecciones y reshape
queries = self.W_query(x).view(b, num_tokens, num_heads, head_dim).transpose(1, 2)
keys = self.W_key(x).view(b, num_tokens, num_heads, head_dim).transpose(1, 2)
values = self.W_value(x).view(b, num_tokens, num_heads, head_dim).transpose(1, 2)

# 2. Aplicar RoPE (solo a Q y K)
if self.use_rope:
    cos, sin = self.rope(num_tokens)
    queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin)

# 3. SDPA con Q y K rotados
context_vec = F.scaled_dot_product_attention(
    queries, keys, values,
    attn_mask=None,
    dropout_p=self.dropout.p if self.training else 0.0,
    is_causal=True
)
```

#### Ventajas vs alternativas

| Técnica | Ventaja | Desventaja |
|---------|---------|------------|
| **Absolute PE** (GPT-2) | Simple | No generaliza a secuencias largas |
| **Learned PE** (BERT) | Flexible | Requiere entrenar, no extrapola |
| **Sinusoidal PE** (Transformer) | Sin parámetros | Posiciones absolutas |
| **RoPE** | Relativo + sin parámetros + extrapola | Más complejo de implementar |
| **ALiBi** | Muy simple | Menos expresivo que RoPE |

#### Modelos que usan RoPE

- LLaMA 2/3
- Mistral
- PaLM
- GPT-NeoX
- Qwen

### RMSNorm vs LayerNorm

**LayerNorm:**
```python
mean = x.mean(dim=-1, keepdim=True)
var = x.var(dim=-1, keepdim=True)
norm_x = (x - mean) / sqrt(var + eps)
return scale * norm_x + shift
```

**RMSNorm (más rápido, sin centrado):**
```python
norm = x.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
return x * norm * weight
```

### Pre-LN vs Post-LN

```python
# Post-LN (GPT-2 original) - menos estable
x = x + attn(x)
x = norm(x)

# Pre-LN (moderno) - más estable
x = x + attn(norm(x))
```

### SwiGLU vs GELU

**GELU clásico:**
```python
x = linear1(x)
x = gelu(x)
x = linear2(x)
```

**SwiGLU (LLaMA):**
```python
x = silu(w1(x)) * w3(x)  # Gate mechanism
x = w2(x)
```

**ReLU² (NanoChat, más simple):**
```python
x = linear1(x)
x = relu(x).square()
x = linear2(x)
```

### QK Norm

Normalizar Q y K después de RoPE, antes del attention score:
```python
q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
q, k = rms_norm(q), rms_norm(k)  # ← QK Norm
attn_scores = q @ k.transpose(-2, -1)
```

**Beneficio:** Estabiliza entrenamiento con secuencias largas.

### Logit Softcap

Evita logits extremos que causan inestabilidad:
```python
softcap = 15
logits = lm_head(x)
logits = softcap * torch.tanh(logits / softcap)
```

---

## 5. Optimizers para LLMs

### AdamW (estándar)

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=4e-4,
    betas=(0.9, 0.95),  # GPT-style
    eps=1e-8,
    weight_decay=0.1
)
```

**Configuración típica:**
- `betas=(0.9, 0.95)` - GPT/NanoChat style (más bajo que default)
- `weight_decay=0.1` - Regularización
- No aplicar weight decay a: norms, biases, embeddings

### Separar param groups

```python
decay_params = []
no_decay_params = []
for name, param in model.named_parameters():
    if 'weight' in name and 'norm' not in name and 'emb' not in name:
        decay_params.append(param)
    else:
        no_decay_params.append(param)

optimizer = AdamW([
    {'params': decay_params, 'weight_decay': 0.1},
    {'params': no_decay_params, 'weight_decay': 0.0}
], lr=lr, betas=(0.9, 0.95))
```

### Muon (experimental, NanoChat)

Optimizer especializado para matrices de transformers:
```python
# AdamW para embeddings
# Muon para linear layers
optimizers = [adamw_opt, muon_opt]
```

---

## 6. Data y Tokenización

### Tokenizers comunes

| Tokenizer | Vocab Size | Usado en |
|-----------|------------|----------|
| GPT-2 BPE | 50,257 | GPT-2/3 |
| SentencePiece | Variable | LLaMA, T5 |
| tiktoken | 100K+ | GPT-4 |

### Métricas de tokenización

- **Compression rate**: Caracteres por token (mayor = mejor)
- **Fertility**: Tokens por palabra

### Datasets comunes para pretraining

| Dataset | Tamaño | Descripción |
|---------|--------|-------------|
| FineWeb-Edu | 1.3T tokens | Web filtrado por calidad educativa |
| The Pile | 800B tokens | Mix diverso (código, libros, web) |
| RedPajama | 1.2T tokens | Réplica abierta de LLaMA data |
| C4 | 156B tokens | Common Crawl limpio |

### Multi-dataset training

**TaskMixture (NanoChat style):**
```python
train_dataset = TaskMixture([
    SmolTalk(split="train"),      # 460K rows
    MMLU(split="train"),          # 100K rows
    GSM8K(split="train"),         # 8K rows
    CustomJSON(filepath=path),    # Custom data
    CustomJSON(filepath=path),    # Repeat for oversampling!
])
```

**Truco:** Para oversamplear un dataset, simplemente pásalo múltiples veces.

---

## 7. Training Recipes

### Pipeline típico de training

```
1. Pretraining (Base model)
   └── Dataset: Web crawl (FineWeb, C4)
   └── Objetivo: Next token prediction
   └── Tokens: ~20x params (Chinchilla)

2. Midtraining (opcional)
   └── Dataset: Mix de alta calidad
   └── Objetivo: Mejorar capabilities específicas

3. SFT (Supervised Fine-tuning)
   └── Dataset: Instrucciones/conversaciones
   └── Objetivo: Seguir instrucciones

4. RLHF / DPO (Alignment)
   └── Dataset: Preferencias humanas
   └── Objetivo: Alinear con valores humanos
```

### Hyperparámetros típicos

| Parámetro | Valor típico | Notas |
|-----------|--------------|-------|
| Learning rate | 1e-4 a 6e-4 | Escala con √(batch_size) |
| Batch size | 0.5M - 4M tokens | Más grande = más estable |
| Weight decay | 0.1 | No aplicar a norms/biases |
| Gradient clip | 1.0 | Previene explosión de gradientes |
| Warmup | 1-5% de steps | Estabiliza inicio |

### Mixed Precision

```python
# bf16 (preferido en Ampere+)
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    logits = model(input_ids)
    loss = F.cross_entropy(...)

# fp16 (requiere GradScaler)
scaler = torch.cuda.amp.GradScaler()
with torch.autocast(device_type='cuda', dtype=torch.float16):
    loss = ...
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## 8. Inference y Serving

### KV Cache

Evita recomputar keys/values de tokens anteriores:
```python
class KVCache:
    def __init__(self, max_seq_len, num_layers, num_heads, head_dim):
        self.k_cache = torch.zeros(num_layers, max_seq_len, num_heads, head_dim)
        self.v_cache = torch.zeros(num_layers, max_seq_len, num_heads, head_dim)
        self.pos = 0
    
    def insert_kv(self, layer_idx, k, v):
        seq_len = k.size(1)
        self.k_cache[layer_idx, self.pos:self.pos+seq_len] = k
        self.v_cache[layer_idx, self.pos:self.pos+seq_len] = v
        return self.k_cache[layer_idx, :self.pos+seq_len], ...
```

### Sampling strategies

```python
# Greedy
next_token = logits.argmax(dim=-1)

# Temperature
logits = logits / temperature
probs = F.softmax(logits, dim=-1)
next_token = torch.multinomial(probs, 1)

# Top-k
top_k_logits, top_k_indices = logits.topk(k)
probs = F.softmax(top_k_logits, dim=-1)
next_token = top_k_indices[torch.multinomial(probs, 1)]

# Top-p (nucleus)
sorted_probs, sorted_indices = probs.sort(descending=True)
cumsum = sorted_probs.cumsum(dim=-1)
mask = cumsum <= p
# ... sample from masked distribution
```

### Optimizaciones de inference

- **FlashAttention**: Fused attention kernel, memory efficient
- **PagedAttention (vLLM)**: Manejo eficiente de KV cache
- **Speculative decoding**: Draft model + verify
- **Quantization**: INT8/INT4 para reducir memoria

---

## 9. Preguntas Comunes de Entrevista

### Arquitectura

**Q: ¿Por qué Pre-LN es mejor que Post-LN?**
> Pre-LN es más estable durante el entrenamiento porque normaliza antes de cada sublayer, evitando gradientes muy grandes. Post-LN puede ser inestable especialmente con modelos profundos.

**Q: ¿Qué ventaja tiene RoPE sobre embeddings posicionales absolutos?**
> RoPE permite mejor generalización a longitudes de secuencia no vistas durante entrenamiento, y decae naturalmente con la distancia entre tokens.

**Q: ¿Por qué se usa RMSNorm en lugar de LayerNorm?**
> RMSNorm es ~10-15% más rápido porque no calcula la media, solo la norma. Empíricamente funciona igual o mejor.

### Training

**Q: ¿Por qué los LLMs se entrenan con pocos epochs?**
> A diferencia de vision donde data augmentation crea variantes infinitas, en texto repetir datos lleva a memorización. Chinchilla demostró que es más eficiente ver más datos únicos.

**Q: ¿Cómo funciona el warmup en LLMs?**
> El warmup aumenta gradualmente el learning rate desde ~0 hasta el máximo durante los primeros N steps. Esto estabiliza el entrenamiento inicial cuando los gradientes son ruidosos.

**Q: ¿Por qué weight decay no se aplica a biases y norms?**
> Weight decay es una forma de regularización L2. Aplicarlo a biases y norms puede interferir con su función de centrar/escalar activaciones.

### Scaling

**Q: Explica la Chinchilla scaling law**
> Para un budget de compute fijo, el número óptimo de tokens de entrenamiento es ~20x el número de parámetros. Esto implica que modelos más pequeños entrenados con más datos pueden superar a modelos más grandes undertrained.

**Q: ¿Cómo escala el costo de compute con el tamaño del modelo?**
> FLOPs ≈ 6 × N × D, donde N = parámetros y D = tokens. El "6" viene de: 2 FLOPs/param para forward, 4 para backward.

### Inference

**Q: ¿Qué es el KV cache y por qué es importante?**
> El KV cache almacena los keys y values computados de tokens anteriores, evitando recomputarlos en cada step de generación. Reduce la complejidad de O(n²) a O(n) por token generado.

**Q: ¿Cuál es la diferencia entre top-k y top-p sampling?**
> Top-k muestrear de los k tokens más probables (fijo). Top-p (nucleus) muestrea de los tokens cuya probabilidad acumulada no excede p (dinámico, se adapta a la distribución).

---

## Referencias

- [Chinchilla Paper](https://arxiv.org/abs/2203.15556) - Training Compute-Optimal LLMs
- [RoFormer (RoPE)](https://arxiv.org/abs/2104.09864) - Rotary Position Embedding
- [LLaMA Paper](https://arxiv.org/abs/2302.13971) - LLaMA architecture
- [GLU Variants (SwiGLU)](https://arxiv.org/abs/2002.05202) - Gated Linear Units
- [RMSNorm](https://arxiv.org/abs/1910.07467) - Root Mean Square Normalization
- [NanoChat](https://github.com/karpathy/nanochat) - Karpathy's minimal ChatGPT
- [NanoGPT](https://github.com/karpathy/nanoGPT) - Karpathy's GPT training code

---

*Última actualización: Diciembre 2025*
