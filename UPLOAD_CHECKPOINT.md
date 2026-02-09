# Cómo subir tu checkpoint local a RunPod

Tenés **3 opciones** para subir tu checkpoint `checkpoint_step_171849.pt` al Network Volume de RunPod.

---

## Opción 1: runpodctl (Recomendada - Más fácil)

### Instalación en Windows

```powershell
# Descargar runpodctl
Invoke-WebRequest -Uri "https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-windows-amd64.exe" -OutFile "runpodctl.exe"

# Mover a una carpeta en PATH (opcional)
Move-Item runpodctl.exe C:\Windows\System32\runpodctl.exe
```

### Configurar API Key

1. Ve a RunPod → Settings → API Keys
2. Crea una nueva API Key
3. Configura runpodctl:

```powershell
runpodctl config --apiKey YOUR_API_KEY_HERE
```

### Subir checkpoint al Network Volume

```powershell
# Sintaxis: runpodctl send <archivo_local> <volume_id>:<ruta_destino>

# Ejemplo: subir checkpoint al volumen
runpodctl send "C:\Users\Usuario\CascadeProjects\windsurf-project\pre-train\Experiment 2 - ChatML + Optimizations\checkpoints\checkpoint_step_171849.pt" YOUR_VOLUME_ID:/Experiment 2 - ChatML + Optimizations/checkpoints/

# Para subir toda la carpeta de checkpoints:
runpodctl send "C:\Users\Usuario\CascadeProjects\windsurf-project\pre-train\Experiment 2 - ChatML + Optimizations\checkpoints" YOUR_VOLUME_ID:/Experiment 2 - ChatML + Optimizations/
```

**Encontrar tu Volume ID:**
- Ve a RunPod → Storage → Network Volumes
- Copia el ID del volumen (ej: `mrt3b0u1x6`)

**Ventajas:**
- ✅ No necesitas un pod corriendo
- ✅ Sube directo al Network Volume
- ✅ Rápido y confiable

---

## Opción 2: S3 API del Network Volume (Avanzado)

Cada Network Volume tiene un endpoint S3. Podés usar `rclone` o AWS CLI.

### Configurar rclone

1. Instalar rclone: https://rclone.org/downloads/

2. Configurar:
```powershell
rclone config
```

3. Crear nuevo remote:
```
n) New remote
name> runpod
Storage> s3
provider> Other
env_auth> false
access_key_id> YOUR_ACCESS_KEY  # De RunPod Settings
secret_access_key> YOUR_SECRET_KEY
region> us-east-1
endpoint> https://s3ap-us-ks-2.runpod.io  # Tu endpoint S3
```

4. Subir checkpoint:
```powershell
rclone copy "C:\Users\Usuario\CascadeProjects\windsurf-project\pre-train\Experiment 2 - ChatML + Optimizations\checkpoints\checkpoint_step_171849.pt" runpod:YOUR_VOLUME_ID/Experiment 2 - ChatML + Optimizations/checkpoints/
```

**Ventajas:**
- ✅ Útil para backups automáticos
- ✅ Soporta sync bidireccional

**Desventajas:**
- ❌ Setup más complejo

---

## Opción 3: SCP/rsync con Pod corriendo (Tradicional)

Requiere tener un pod activo (pagas por GPU mientras subes).

### Paso 1: Configurar SSH Key

```powershell
# Generar key si no tienes
ssh-keygen -t ed25519 -C "tu_email@example.com"

# Ver tu public key
cat C:\Users\Usuario\.ssh\id_ed25519.pub
```

### Paso 2: Agregar key en RunPod

1. Ve a tu Pod → Connect → SSH
2. Pega tu public key
3. Copia el comando SSH (ej: `ssh root@216.81.245.125 -p 17598`)

### Paso 3: Subir con SCP

```powershell
# Sintaxis: scp -P <puerto> <archivo_local> root@<ip>:<ruta_destino>

scp -P 17598 "C:\Users\Usuario\CascadeProjects\windsurf-project\pre-train\Experiment 2 - ChatML + Optimizations\checkpoints\checkpoint_step_171849.pt" root@216.81.245.125:/workspace/Experiment 2 - ChatML + Optimizations/checkpoints/
```

**Ventajas:**
- ✅ Familiar si ya usas SSH

**Desventajas:**
- ❌ Necesitas pod corriendo (pagas GPU)
- ❌ Más lento que runpodctl

---

## 🎯 Recomendación

**Usa runpodctl (Opción 1)** porque:
- No pagas GPU mientras subes
- Más rápido
- Más simple

---

## 📋 Checklist después de subir

1. **Verificar que el checkpoint llegó:**
   ```bash
   # En el pod (Web Terminal)
   ls -lh /workspace/"Experiment 2 - ChatML + Optimizations"/checkpoints/
   ```

2. **Verificar tamaño del archivo:**
   ```bash
   # Debe ser ~800MB (depende de tu modelo)
   du -h /workspace/"Experiment 2 - ChatML + Optimizations"/checkpoints/checkpoint_step_171849.pt
   ```

3. **Verificar que el YAML apunta al checkpoint:**
   ```yaml
   storage:
     checkpoint_to_resume: checkpoint_step_171849.pt
   ```

4. **Probar que carga correctamente:**
   ```bash
   cd /workspace/windsurf-project
   python -c "
   import torch
   ckpt = torch.load('/workspace/Experiment 2 - ChatML + Optimizations/checkpoints/checkpoint_step_171849.pt', map_location='cpu')
   print('✓ Checkpoint válido')
   print(f'  Global step: {ckpt[\"global_step\"]}')
   print(f'  Val loss: {ckpt.get(\"val_loss\", \"N/A\")}')
   "
   ```

---

## 💡 Tips

### Subir solo el checkpoint necesario
No subas todos los checkpoints viejos, solo el que vas a usar:
```powershell
runpodctl send checkpoint_step_171849.pt VOLUME_ID:/Experiment 2 - ChatML + Optimizations/checkpoints/
```

### Comprimir antes de subir (opcional)
Si la conexión es lenta:
```powershell
# Comprimir
tar -czf checkpoint_171849.tar.gz checkpoint_step_171849.pt

# Subir
runpodctl send checkpoint_171849.tar.gz VOLUME_ID:/

# Descomprimir en el pod
tar -xzf /workspace/checkpoint_171849.tar.gz -C /workspace/"Experiment 2 - ChatML + Optimizations"/checkpoints/
```

### Verificar integridad con hash
```powershell
# En Windows
certutil -hashfile checkpoint_step_171849.pt MD5

# En RunPod
md5sum /workspace/"Experiment 2 - ChatML + Optimizations"/checkpoints/checkpoint_step_171849.pt
```

Los hashes deben coincidir ✅

---

## 🚨 Troubleshooting

### Error: "Permission denied"
```bash
# En el pod, verificar permisos
ls -la /workspace/"Experiment 2 - ChatML + Optimizations"/checkpoints/

# Arreglar permisos si es necesario
chmod 644 /workspace/"Experiment 2 - ChatML + Optimizations"/checkpoints/*.pt
```

### Error: "No space left on device"
El volumen está lleno. Opciones:
1. Eliminar checkpoints viejos
2. Aumentar tamaño del Network Volume en RunPod

### Checkpoint no se encuentra después de subir
Verificar la ruta exacta:
```bash
find /workspace -name "checkpoint_step_171849.pt"
```

---

## 📊 Ejemplo completo: Workflow de subida

```powershell
# 1. Configurar runpodctl (solo una vez)
runpodctl config --apiKey sk-xxxxxxxxxxxxx

# 2. Subir checkpoint
runpodctl send "C:\Users\Usuario\CascadeProjects\windsurf-project\pre-train\Experiment 2 - ChatML + Optimizations\checkpoints\checkpoint_step_171849.pt" mrt3b0u1x6:/Experiment 2 - ChatML + Optimizations/checkpoints/

# 3. Deploy pod con ese Network Volume

# 4. En el pod, verificar
ls -lh /workspace/"Experiment 2 - ChatML + Optimizations"/checkpoints/

# 5. Ejecutar training
cd /workspace/windsurf-project
./setup_and_train.sh pre-train/experiments/Experiment2-ChatML-Optimizations-H100.yaml

# El training debe reanudar desde step 171850 ✅
```
