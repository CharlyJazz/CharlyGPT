"""
MLflow integration para visualizar experimentos de pre-training GPT-2
Testea todos los checkpoints encontrados en la carpeta del experimento.

Uso:
    python mlflow_viewer.py --config experiments/SmallGPT2-Samples2M.yaml --questions 5
"""

import mlflow
import pandas as pd
import torch
import sys
import os
import yaml
import argparse
from pathlib import Path
from glob import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arch.gpt_model import GPTModel
from arch.config import GPT_CONFIG_124M
from utils import tokenizer, text_to_token_ids, token_ids_to_text, generate_text_simple
from datasets import load_dataset

# MLflow config
MLFLOW_TRACKING_URI = "file:./mlruns"


def load_experiment_config(config_path: str) -> dict:
    """Carga la configuración del experimento desde YAML"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    experiment_name = config.get("experiment_name", "default_experiment")
    base_folder = config.get("storage", {}).get("base_folder", ".")
    checkpoints_dir = os.path.join(base_folder, experiment_name, "checkpoints")
    
    return {
        "experiment_name": experiment_name,
        "base_folder": base_folder,
        "checkpoints_dir": checkpoints_dir,
    }


def get_existing_runs(experiment_name: str) -> set:
    """Obtiene los steps ya registrados en MLflow para evitar duplicados"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            return set()
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="",
        )
        
        if runs.empty:
            return set()
        
        # Extraer steps de los nombres de los runs
        existing_steps = set()
        for run_name in runs['tags.mlflow.runName'].dropna():
            if run_name.startswith("step_"):
                try:
                    step = int(run_name.replace("step_", ""))
                    existing_steps.add(step)
                except ValueError:
                    pass
        
        return existing_steps
    except Exception as e:
        print(f"[WARN] Error obteniendo runs existentes: {e}")
        return set()


def setup_mlflow(experiment_name: str):
    """Configura MLflow"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)
    print(f"[OK] MLflow configurado")
    print(f"  - Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"  - Experiment: {experiment_name}")


def view_mlflow_ui():
    """Instrucciones para ver MLflow UI"""
    print("\n" + "="*60)
    print("PARA VER MLFLOW UI:")
    print("="*60)
    print("\n1. Abre una terminal en esta carpeta")
    print("2. Ejecuta: python -m mlflow ui")
    print("3. Abre en navegador: http://localhost:5000")
    print("\n" + "="*60)


def list_available_checkpoints(checkpoints_dir: str):
    """Lista todos los checkpoints disponibles en una carpeta"""
    checkpoints = glob(os.path.join(checkpoints_dir, "checkpoint_step_*.pt"))
    
    def get_step_from_name(path):
        name = Path(path).stem
        if name.startswith("checkpoint_step_"):
            return int(name.split("_")[-1])
        return 0  # best_model u otros
    
    checkpoints.sort(key=get_step_from_name)
    
    # También buscar best_model.pt (dentro de checkpoints)
    best_model = os.path.join(checkpoints_dir, "best_model.pt")
    if os.path.exists(best_model):
        checkpoints.append(best_model)
    
    return checkpoints


def load_checkpoint_info(checkpoint_path: str) -> dict:
    """Carga info de un checkpoint sin cargar el modelo completo"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return {
        "path": checkpoint_path,
        "name": Path(checkpoint_path).stem,
        "step": checkpoint.get('global_step', 0),
        "epoch": checkpoint.get('epoch', 0),
        "val_loss": checkpoint.get('val_loss', None),
        "train_loss": checkpoint.get('train_loss', None),
        "best_val_loss": checkpoint.get('best_val_loss', None),
    }


def generate_response(model, prompt: str, device: str = "cuda", max_new_tokens: int = 150):
    """Genera una respuesta dado un prompt"""
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(prompt, tokenizer).to(device)
    
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=max_new_tokens,
            context_size=context_size,
            temperature=0.7,
            top_k=50
        )
    
    return token_ids_to_text(token_ids, tokenizer)


def evaluate_checkpoint(checkpoint_path: str, test_questions: list, device: str = "cuda") -> dict:
    """
    Evalúa un checkpoint con preguntas de prueba.
    Retorna métricas y respuestas.
    """
    # Cargar modelo
    model = GPTModel(GPT_CONFIG_124M)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    info = load_checkpoint_info(checkpoint_path)
    
    results = []
    for i, q in enumerate(test_questions):
        query = q['query']
        prompt = f"Q: {query}\n\nA:"
        
        full_response = generate_response(model, prompt, device, max_new_tokens=200)
        
        if "A:" in full_response:
            model_answer = full_response.split("A:")[-1].strip()
        else:
            model_answer = full_response[len(prompt):].strip()
        
        results.append({
            'question_num': i + 1,
            'query': query,
            'model_answer': model_answer,
            'dataset_answer': q['synthetic_answer'],
        })
    
    # Limpiar memoria
    del model
    torch.cuda.empty_cache()
    
    return {
        'info': info,
        'results': results,
    }


def test_all_checkpoints(config: dict, num_questions: int = 5):
    """
    Testea TODOS los checkpoints encontrados en la carpeta del experimento.
    Es IDEMPOTENTE: no registra checkpoints que ya existen en MLflow.
    
    Args:
        config: Configuración cargada desde YAML
        num_questions: Número de preguntas de prueba
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    experiment_name = config["experiment_name"]
    checkpoints_dir = config["checkpoints_dir"]
    
    setup_mlflow(experiment_name)
    
    # Obtener runs existentes para idempotencia
    existing_steps = get_existing_runs(experiment_name)
    if existing_steps:
        print(f"[INFO] Steps ya registrados en MLflow: {sorted(existing_steps)}")
    
    # 1. Listar checkpoints
    checkpoints = list_available_checkpoints(checkpoints_dir)
    print(f"\n{'='*60}")
    print(f"CHECKPOINTS ENCONTRADOS: {len(checkpoints)}")
    print(f"{'='*60}")
    
    new_checkpoints = []
    for cp in checkpoints:
        info = load_checkpoint_info(cp)
        status = "[SKIP]" if info['step'] in existing_steps else "[NEW]"
        print(f"  - Step {info['step']:,} | Val Loss: {info['val_loss']} | {status}")
        if info['step'] not in existing_steps:
            new_checkpoints.append(cp)
    
    if not new_checkpoints:
        print(f"\n[OK] Todos los checkpoints ya están registrados. Nada que hacer.")
        view_mlflow_ui()
        return
    
    print(f"\nCheckpoints nuevos a evaluar: {len(new_checkpoints)}")
    
    # 2. Obtener preguntas de prueba
    print(f"\n{'='*60}")
    print("OBTENIENDO PREGUNTAS DE PRUEBA")
    print(f"{'='*60}")
    
    dataset = load_dataset("PleIAs/SYNTH", split="train", streaming=True)
    test_questions = []
    for item in dataset:
        if item.get('language') == 'en':
            test_questions.append(item)
            if len(test_questions) >= num_questions:
                break
    print(f"[OK] {len(test_questions)} preguntas obtenidas")
    
    # 3. Evaluar cada checkpoint NUEVO y registrar en MLflow
    print(f"\n{'='*60}")
    print("EVALUANDO CHECKPOINTS")
    print(f"{'='*60}")
    
    for checkpoint_path in new_checkpoints:
        info = load_checkpoint_info(checkpoint_path)
        run_name = f"step_{info['step']}"
        
        print(f"\n--- Evaluando: {run_name} ---")
        
        with mlflow.start_run(run_name=run_name):
            # Log parámetros
            mlflow.log_params({
                "checkpoint_name": info['name'],
                "step": info['step'],
                "epoch": info['epoch'],
            })
            
            # Log métricas del checkpoint
            if info['val_loss'] is not None:
                mlflow.log_metric("val_loss", info['val_loss'])
            if info['train_loss'] is not None:
                mlflow.log_metric("train_loss", info['train_loss'])
            
            # Evaluar
            eval_results = evaluate_checkpoint(checkpoint_path, test_questions, device)
            
            # Log respuestas como tabla
            df = pd.DataFrame(eval_results['results'])
            mlflow.log_table(df, artifact_file="responses.json")
            
            # Log métricas de respuestas
            avg_len = df['model_answer'].str.len().mean()
            mlflow.log_metric("avg_response_length", avg_len)
            
            # Guardar CSV
            csv_path = f"checkpoint_{info['step']}_responses.csv"
            df.to_csv(csv_path, index=False)
            mlflow.log_artifact(csv_path)
            os.remove(csv_path)  # Limpiar archivo temporal
            
            print(f"  [OK] Registrado en MLflow | Val Loss: {info['val_loss']}")
    
    print(f"\n{'='*60}")
    print("[OK] TESTING COMPLETADO")
    print(f"{'='*60}")
    view_mlflow_ui()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MLflow Checkpoint Tester - Testea todos los checkpoints de un experimento",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplo:
  python mlflow_viewer.py --config experiments/SmallGPT2-Samples2M.yaml --questions 5
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Ruta al archivo YAML de configuración del experimento"
    )
    
    parser.add_argument(
        "--questions", "-q",
        type=int,
        default=5,
        help="Número de preguntas de prueba (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Cargar configuración desde YAML
    config = load_experiment_config(args.config)
    
    print("\n" + "="*60)
    print("MLFLOW CHECKPOINT TESTER (IDEMPOTENTE)")
    print("="*60)
    print(f"Config: {args.config}")
    print(f"Experimento: {config['experiment_name']}")
    print(f"Checkpoints: {config['checkpoints_dir']}")
    print(f"Preguntas: {args.questions}")
    
    # Testear todos los checkpoints
    test_all_checkpoints(config, num_questions=args.questions)
