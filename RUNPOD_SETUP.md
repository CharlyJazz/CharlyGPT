# RunPod Training Setup Guide

Este documento explica cómo ejecutar el entrenamiento en RunPod usando el script automatizado.

## 🚀 Quick Start

### 1. Deploy el Pod en RunPod

- GPU: A100 (80GB) o H100
- Network Volume: Montado en `/workspace`
- Habilitar **Web Terminal**

### 2. Clonar el repositorio

```bash
cd /workspace
git clone https://github.com/CharlyJazz/windsurf-project.git
cd windsurf-project  # O el nombre que tenga tu repo
```

### 3. Ejecutar el script de setup

```bash
chmod +x setup_and_train.sh
./setup_and_train.sh
```

¡Eso es todo! El script se encarga de:
- ✅ Verificar Python y CUDA
- ✅ Instalar dependencias faltantes (PyTorch, tiktoken, etc.)
- ✅ Configurar variables de entorno
- ✅ Ejecutar el training

---

## 📋 Uso Avanzado

### Especificar un experimento diferente

```bash
./setup_and_train.sh pre-train/experiments/MiExperimento.yaml
```

### Ver logs en tiempo real

El script muestra todo el output del training. Para detener gracefully:

```bash
# Presiona Ctrl+C una vez
# El script guardará un checkpoint de emergencia y saldrá limpiamente
```

### Reanudar training desde checkpoint

El script detecta automáticamente si hay un checkpoint configurado en el YAML:

```yaml
storage:
  checkpoint_to_resume: checkpoint_step_5000.pt  # Se reanuda desde aquí
```

---

## 🔧 Configuración del YAML para RunPod

Asegúrate de que tu YAML tenga estos valores para RunPod:

```yaml
experiment_name: "Experiment 2 - ChatML + Optimizations"

data:
  num_samples: null  # null = streaming ilimitado
  local_dataset_path: null  # null = usar HuggingFace Hub

storage:
  base_folder: /workspace  # ⚠️ IMPORTANTE: usar /workspace para persistencia
  checkpoint_to_resume: null  # o nombre del checkpoint para reanudar

training:
  total_steps: 200000  # Ajustar según presupuesto
```

---

## 📊 Verificar que el checkpoint se guardó

Después de que el training guarde un checkpoint:

```bash
ls -lh /workspace/"Experiment 2 - ChatML + Optimizations"/checkpoints/
```

Deberías ver archivos `.pt` como:
```
checkpoint_step_1000.pt
checkpoint_step_2000.pt
best_model.pt
```

---

## 🔄 Smoke Test (Prueba Rápida)

Para verificar que todo funciona antes de un training largo:

### 1. Editar el YAML temporalmente

```yaml
training:
  total_steps: 100  # Solo 100 steps para probar
  save_every_n_iterations: 50  # Guardar cada 50 steps
```

### 2. Ejecutar el script

```bash
./setup_and_train.sh
```

### 3. Verificar checkpoint

```bash
ls -lh /workspace/"Experiment 2 - ChatML + Optimizations"/checkpoints/
```

### 4. Terminar el pod

En la UI de RunPod: **Terminate Pod**

### 5. Recrear el pod con el mismo Network Volume

Deploy nuevo pod → Seleccionar el mismo Network Volume

### 6. Verificar persistencia

```bash
cd /workspace/windsurf-project
ls -lh /workspace/"Experiment 2 - ChatML + Optimizations"/checkpoints/
# Los checkpoints deben seguir ahí ✅
```

### 7. Reanudar training

Editar el YAML:
```yaml
storage:
  checkpoint_to_resume: checkpoint_step_50.pt  # Último checkpoint guardado
```

Ejecutar:
```bash
./setup_and_train.sh
```

El training debe reanudar desde el step 51 ✅

---

## 🐛 Troubleshooting

### Error: "CUDA not available"

```bash
# Verificar que CUDA esté disponible
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

Si devuelve `False`, reinstalar PyTorch:
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Error: "Checkpoint file not found"

El checkpoint especificado en `checkpoint_to_resume` no existe. Opciones:

1. Poner `checkpoint_to_resume: null` para empezar de cero
2. Verificar el nombre exacto del checkpoint:
   ```bash
   ls /workspace/"Experiment 2 - ChatML + Optimizations"/checkpoints/
   ```

### Error: "No space left on device"

El volumen está lleno. Opciones:

1. Eliminar checkpoints viejos (el script guarda solo los últimos 3)
2. Aumentar el tamaño del Network Volume en RunPod

### Training muy lento

Verificar que esté usando GPU:
```bash
# Durante el training, en otra terminal:
nvidia-smi
# Debe mostrar uso de GPU y memoria
```

---

## 💰 Costos de RunPod

### Durante training (Pod Running)
- **A100 (80GB)**: ~$1.50/hr
- **H100**: ~$3.00/hr

### Network Volume (siempre activo)
- **100 GB**: ~$0.01/hr = $0.24/día = $7/mes

### Recomendación
- Terminar el pod cuando no estés entrenando
- El Network Volume conserva todos los checkpoints
- Recrear el pod cuando quieras continuar

---

## 📝 Checklist Pre-Training

Antes de un training largo (ej: 200k steps en H100):

- [ ] Smoke test completado exitosamente
- [ ] Checkpoint persistence verificada
- [ ] Resume functionality probada
- [ ] YAML configurado con `base_folder: /workspace`
- [ ] `total_steps` ajustado según presupuesto
- [ ] Presupuesto de RunPod suficiente
- [ ] Monitoring configurado (opcional: MLflow, Weights & Biases)

---

## 🎯 Ejemplo Completo: Training en H100

```bash
# 1. Deploy H100 pod con Network Volume en /workspace
# 2. En el Web Terminal:

cd /workspace
git clone https://github.com/CharlyJazz/windsurf-project.git
cd windsurf-project

# 3. Ejecutar training
./setup_and_train.sh

# 4. Monitorear progreso
# El script muestra loss, LR, y samples cada 10 steps
# Checkpoints se guardan cada 1000 steps (configurable en YAML)

# 5. Para detener gracefully: Ctrl+C
# 6. Para terminar el pod: UI de RunPod → Terminate
```

---

## 📚 Recursos Adicionales

- **MLflow UI**: Para ver métricas históricas
  ```bash
  cd /workspace/windsurf-project/pre-train
  python -m mlflow ui --host 0.0.0.0 --port 5000
  # Acceder desde RunPod HTTP Services
  ```

- **Dataset Diagnostics**: Logs detallados en
  ```
  /workspace/"Experiment 2 - ChatML + Optimizations"/diagnostics/
  ```

- **Logs de training**: Stdout del script (considerar usar `tee` para guardar)
  ```bash
  ./setup_and_train.sh 2>&1 | tee training.log
  ```
