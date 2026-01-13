# Experiment 2 - Checklist de Implementación

## 📋 Resumen del Experimento
**Objetivo**: Migrar arquitectura GPT-2 actual hacia diseño moderno (LLaMA/NanoChat style) con formato ChatML

**Configuración Base**:
- 24 layers (aumentado de 12)
- 768 emb_dim
- 12 heads
- Dropout dinámico: 0.0 → 0.05
- 2M samples del SYNTH dataset
- Formato ChatML con tokens especiales

---

## FASE 1: Quick Wins (Sin cambiar arquitectura)
**Objetivo**: Estabilidad + velocidad sin cambiar comportamiento del modelo

### ✅ Tareas Fase 1

- [x] **1.1 GELU Nativo**
  - Archivo: `arch/feed_forward.py`
  - Cambio: Reemplazar `from arch.gelu import GELU` → `nn.GELU(approximate="tanh")`
  - Beneficio: +velocidad, elimina bugs de device mismatch
  - Riesgo: Ninguno

- [x] **1.2 SDPA (Scaled Dot-Product Attention)**
  - Archivo: `arch/attention.py`
  - Cambio: Reemplazar attention manual por `F.scaled_dot_product_attention`
  - Código:
    ```python
    context_vec = F.scaled_dot_product_attention(
        queries, keys, values,
        attn_mask=None,
        dropout_p=self.dropout.p if self.training else 0.0,
        is_causal=True
    ).transpose(1, 2)
    ```
  - Beneficio: +velocidad, -memoria, aprovecha FlashAttention
  - Riesgo: Bajo

- [x] **1.3 Weight Tying**
  - Archivo: `arch/gpt_model.py`
  - Cambio: En `__init__`, después de crear `out_head`:
    ```python
    self.out_head.weight = self.tok_emb.weight
    ```
  - Beneficio: -38M params, mejor perplexity
  - Riesgo: Ninguno

- [x] **1.4 Sanity Check Fase 1**
  - Correr training ~500 steps
  - Verificar: sin NaN, loss bajando, tokens/sec mejorado

---

## FASE 2: RoPE (Rotary Position Embeddings)
**Objetivo**: Mejor generalización posicional (estándar LLaMA/Mistral)

### ✅ Tareas Fase 2

- [x] **2.1 Crear arch/rope.py**
  - Implementar clase `RotaryEmbedding`
  - Implementar función `rotate_half`
  - Implementar función `apply_rotary_pos_emb`
  - Código completo en líneas 145-181 del proposal

- [x] **2.2 Modificar arch/attention.py**
  - Importar RoPE functions
  - Aplicar RoPE a Q y K después de reshape
  - Mantener SDPA de Fase 1

- [x] **2.3 Modificar arch/gpt_model.py**
  - Remover/comentar `self.pos_emb = nn.Embedding(...)`
  - Crear instancia de RoPE
  - Pasar RoPE a los transformer blocks

- [x] **2.4 Actualizar config.yaml**
  - Agregar parámetros:
    ```yaml
    use_rope: true
    rope_base: 10000
    ```

- [x] **2.5 Sanity Check Fase 2**
  - Correr training ~500 steps
  - Verificar: sin NaN, loss bajando
  - Probar generalización a seq_len > 512

---

## FASE 2.5: NanoChat Additions
**Objetivo**: Técnicas específicas de NanoChat para estabilidad

### ✅ Tareas Fase 2.5

- [ ] **2.5.1 QK Norm**
  - Archivo: `arch/attention.py`
  - Agregar normalización a Q y K después de RoPE:
    ```python
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    q = F.rms_norm(q, (q.size(-1),))
    k = F.rms_norm(k, (k.size(-1),))
    ```
  - Beneficio: Estabiliza training con secuencias largas

- [ ] **2.5.2 Norm después de Embedding**
  - Archivo: `arch/gpt_model.py`
  - En `forward`, después de `tok_emb`:
    ```python
    x = self.tok_emb(idx)
    x = F.rms_norm(x, (x.size(-1),))
    ```

- [ ] **2.5.3 Logit Softcap**
  - Archivo: `arch/gpt_model.py`
  - En `forward`, antes de retornar logits:
    ```python
    softcap = 15.0
    logits = logits.float()
    logits = softcap * torch.tanh(logits / softcap)
    ```
  - Beneficio: Evita logits extremos, estabilidad numérica

- [ ] **2.5.4 Sanity Check Fase 2.5**
  - Correr training ~500 steps
  - Verificar estabilidad mejorada

---

## FASE 3: RMSNorm + Activation
**Objetivo**: Mejoras de calidad (LLaMA/NanoChat style)

### Opción A: LLaMA-style (más complejo)

- [ ] **3A.1 Crear arch/rmsnorm.py**
  - Implementar `RMSNorm` con parámetros aprendibles
  - Código en líneas 196-209 del proposal

- [ ] **3A.2 Crear SwiGLU MLP**
  - Archivo: `arch/feed_forward.py`
  - Implementar `SwiGLUFeedForward`
  - Código en líneas 212-230 del proposal

- [ ] **3A.3 Reemplazar LayerNorm → RMSNorm**
  - Archivos: `arch/transformer_block.py`, `arch/gpt_model.py`
  - Cambiar todas las instancias de LayerNorm

### Opción B: NanoChat-style (más simple) ⭐ RECOMENDADO

- [ ] **3B.1 RMSNorm Funcional**
  - No crear archivo nuevo
  - Usar directamente: `F.rms_norm(x, (x.size(-1),))`
  - Sin parámetros aprendibles

- [ ] **3B.2 ReLU² MLP**
  - Archivo: `arch/feed_forward.py`
  - Cambiar activation:
    ```python
    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()  # ReLU²
        x = self.c_proj(x)
        return x
    ```

- [ ] **3B.3 Actualizar transformer_block.py**
  - Reemplazar LayerNorm por RMSNorm funcional

- [ ] **3B.4 Sanity Check Fase 3**
  - Correr training ~500 steps
  - Comparar loss vs Fase 2

---

## FASE 4: Training Recipe Moderno
**Objetivo**: Hiperparámetros y técnicas SOTA

### ✅ Tareas Fase 4

- [ ] **4.1 AdamW con Parameter Groups**
  - Archivo: Training script
  - Separar params con/sin weight decay:
    ```python
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
    ], lr=learning_rate, betas=(0.8, 0.95), eps=1e-10)
    ```

- [ ] **4.2 Actualizar Betas de AdamW**
  - Cambiar de `(0.9, 0.95)` → `(0.8, 0.95)` (NanoChat style)
  - Cambiar `eps` de `1e-8` → `1e-10`

- [ ] **4.3 Zero Initialization**
  - Archivo: `arch/gpt_model.py`
  - En `__init__`, después de crear layers:
    ```python
    torch.nn.init.zeros_(self.out_head.weight)
    for block in self.trf_blocks:
        torch.nn.init.zeros_(block.mlp.c_proj.weight)
        torch.nn.init.zeros_(block.attn.out_proj.weight)
    ```

- [ ] **4.4 Mixed Precision (bf16)**
  - Archivo: Training script
  - Si GPU soporta bf16 (Ampere+):
    ```python
    dtype = torch.bfloat16
    with torch.autocast(device_type='cuda', dtype=dtype):
        logits = model(input_ids)
        loss = F.cross_entropy(...)
    ```

- [ ] **4.5 Dropout Schedule Dinámico**
  - Ya configurado en YAML:
    ```yaml
    dropout_schedule:
      enabled: true
      initial_dropout: 0.0
      final_dropout: 0.05
      transition_at_progress: 0.6
    ```
  - Implementar lógica en training loop

---

## CONFIGURACIÓN: Actualizar YAML

### ✅ Tareas de Configuración

- [ ] **Config 1: Arquitectura Base**
  - Verificar en `Experiment2-ChatML-Optimizations.yaml`:
    ```yaml
    model:
      n_layers: 24  # Aumentado de 12
      drop_rate: 0.0  # Inicial
      use_native_gelu: true
      use_native_sdpa: true
      use_weight_tying: false  # Cambiar a true en Fase 1
    ```

- [ ] **Config 2: RoPE**
  - Agregar después de Fase 2:
    ```yaml
    use_rope: true
    rope_base: 10000
    ```

- [ ] **Config 3: Normalization**
  - Agregar después de Fase 3:
    ```yaml
    use_rmsnorm: true  # o false si usas funcional
    use_swiglu: false  # true para LLaMA, false para ReLU²
    ```

- [ ] **Config 4: ChatML Tokens**
  - Ya configurado:
    ```yaml
    use_chatml_format: true
    chat_ml_tokens:
      im_start: <|im_start|>
      im_end: <|im_end|>
      think_start: <think>
      think_end: </think>
    ```

- [ ] **Config 5: Training Params**
  - Verificar:
    ```yaml
    training:
      batch_size: 16
      learning_rate: 0.0004
      weight_decay: 0.1
      betas: [0.8, 0.95]  # NanoChat style
      eps: 1.0e-10
      dropout_schedule:
        enabled: true
        initial_dropout: 0.0
        final_dropout: 0.05
        transition_at_progress: 0.6
    ```

---

## DATASET: Preparación ChatML

### ✅ Tareas de Dataset

- [ ] **Dataset 1: Agregar Tokens Especiales**
  - Archivo: Tokenizer config
  - Agregar a vocabulario:
    - `<|im_start|>`
    - `<|im_end|>`
    - `<think>`
    - `</think>`

- [ ] **Dataset 2: Actualizar map_synth**
  - Archivo: Dataset processing script
  - Implementar formato ChatML:
    ```python
    def map_synth(x):
        constraints = f"\n{x['constraints']}" if 'rag' in x['exercise'] else ''
        return {
            "text": (
                f"<|im_start|>user\n{x['query']}{constraints}<|im_end|>\n"
                f"<|im_start|>assistant\n<think>{x['reasoning']}</think>\n"
                f"{x['answer']}<|im_end|>"
            )
        }
    ```

- [ ] **Dataset 3: Verificar Constraints en RAG**
  - Asegurar que ejercicios tipo 'rag' incluyan constraints
  - Tip de @mkurman88: "Remember to include constraints if the exercise is rag"

- [ ] **Dataset 4: Validar Formato**
  - Revisar samples del dataset procesado
  - Verificar que tokens especiales estén correctos
  - **IMPORTANTE**: Recordar que `
