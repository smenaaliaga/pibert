"""
Script mejorado para subir PIBot Joint BERT a Hugging Face Hub de forma reproducible.

Este script crea un checkpoint completamente auto-contenido que puede ser:
1. Cargado localmente con: PIBotPredictor("path")
2. Descargado y usado directamente desde HF
3. Reproducible en cualquier máquina sin dependencias externas

Uso:
    python util/upload_to_hf_improved.py --repo_name smenaaliaga/pibert --model_dir model_out/pibot_model_v5_beto_crf --private
"""

import argparse
import shutil
from pathlib import Path
import torch
from huggingface_hub import HfApi, create_repo


def _infer_label_files_from_args(train_args):
    label_arg_names = [
        "calc_mode_label_file",
        "activity_label_file",
        "region_label_file",
        "investment_label_file",
        "req_form_label_file",
        "slot_label_file",
        "indicator_label_file",
        "metric_type_label_file",
        "seasonal_label_file",
        "frequency_label_file",
        "context_mode_label_file",
        "intent_label_file",
    ]

    label_files = []
    for attr in label_arg_names:
        value = getattr(train_args, attr, None)
        if isinstance(value, str) and value.endswith("_label.txt") and value not in label_files:
            label_files.append(value)

    return label_files


def _build_model_readme(train_args, label_files):
    task = getattr(train_args, "task", "unknown")
    model_name = getattr(train_args, "model_name_or_path", "unknown")
    model_type = getattr(train_args, "model_type", "unknown")

    head_labels = [f for f in label_files if f != "slot_label.txt"]
    head_bullets = "\n".join([f"- `{name.replace('_label.txt', '')}`" for name in head_labels]) or "- N/A"

    return f"""# PIBot Joint BERT

Modelo JointBERT multi-cabeza para consultas económicas en español.

## Configuración del checkpoint

- **Task**: `{task}`
- **Base model**: `{model_name}`
- **Model type**: `{model_type}`

## Cabezas de intención incluidas

{head_bullets}

## Slot filling

- `slot_label.txt` para etiquetado BIO por token.

## Archivos importantes

- `model.safetensors` o `pytorch_model.bin`
- `config.json`
- `training_args.bin`
- tokenizer (`tokenizer_config.json`, `tokenizer.json` y/o vocabulario)
- `modeling_jointbert.py`, `module.py`, `__init__.py`
- `*_label.txt`

## Carga recomendada

```python
from transformers import AutoConfig, AutoTokenizer
from model_in import JointBERT

model_dir = "<ruta_local_o_repo_hf>"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
config = AutoConfig.from_pretrained(model_dir)

# Cargar training_args.bin y labels antes de instanciar JointBERT
```
"""


def prepare_checkpoint_for_hf(model_dir: str, output_dir: str = None, label_dir: str = None, code_dir: str = None):
    """
    Prepara el checkpoint para ser auto-contenido en HF.
    
    Asegura que contiene:
    - model.safetensors o pytorch_model.bin (pesos)
    - config.json (configuración)
    - tokenizer (tokenizer_config.json y tokenizer.json/vocab)
    - training_args.bin (parámetros de entrenamiento)
    - *_label.txt (labels de cada cabeza y slot)
    - modeling_jointbert.py + module.py + __init__.py (código custom)
    - README.md (documentación)
    """
    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {model_dir}")
    
    if output_dir is None:
        output_dir = model_path
    else:
        output_dir = Path(output_dir)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.copytree(model_path, output_dir)
    
    output_dir = Path(output_dir)
    
    weight_files = ["model.safetensors", "pytorch_model.bin"]
    if not any((output_dir / f).exists() for f in weight_files):
        raise FileNotFoundError(f"Falta archivo de pesos ({weight_files}) en {output_dir}")

    required_files = ["config.json", "tokenizer_config.json", "training_args.bin"]
    missing_required = [f for f in required_files if not (output_dir / f).exists()]
    if missing_required:
        raise FileNotFoundError(f"Archivos faltantes en checkpoint: {missing_required}")

    if not (output_dir / "tokenizer.json").exists() and not (output_dir / "vocab.txt").exists():
        raise FileNotFoundError("Falta tokenizer: se requiere tokenizer.json o vocab.txt")

    train_args = torch.load(output_dir / "training_args.bin", weights_only=False)

    project_root = Path(__file__).resolve().parents[1]
    default_label_dir = project_root / "data" / getattr(train_args, "task", "")
    source_label_dir = Path(label_dir) if label_dir else default_label_dir

    inferred_label_files = _infer_label_files_from_args(train_args)
    label_files = [
        name for name in inferred_label_files
        if (output_dir / name).exists() or (source_label_dir / name).exists()
    ]

    if not label_files:
        label_files = sorted([p.name for p in output_dir.glob("*_label.txt")])

    if not label_files and source_label_dir.exists():
        label_files = sorted([p.name for p in source_label_dir.glob("*_label.txt")])

    if not label_files:
        raise FileNotFoundError("No se pudieron inferir archivos *_label.txt desde training_args.bin, model_dir o data/{task}")

    missing_labels = [f for f in label_files if not (output_dir / f).exists()]
    if missing_labels and source_label_dir.exists():
        for name in missing_labels:
            src = source_label_dir / name
            if src.exists():
                shutil.copyfile(src, output_dir / name)

    missing_labels = [f for f in label_files if not (output_dir / f).exists()]
    if missing_labels:
        raise FileNotFoundError(
            f"Archivos de labels faltantes: {missing_labels}. "
            f"Fuente intentada: {source_label_dir}"
        )

    code_files = ["modeling_jointbert.py", "module.py", "__init__.py"]
    default_code_dir = project_root / "model_in"
    source_code_dir = Path(code_dir) if code_dir else default_code_dir

    missing_code = [f for f in code_files if not (output_dir / f).exists()]
    if missing_code and source_code_dir.exists():
        for name in missing_code:
            src = source_code_dir / name
            if src.exists():
                shutil.copyfile(src, output_dir / name)

    missing_code = [f for f in code_files if not (output_dir / f).exists()]
    if missing_code:
        raise FileNotFoundError(
            f"Código custom faltante: {missing_code}. "
            f"Fuente intentada: {source_code_dir}"
        )
    
    # Crear README si no existe
    readme_path = output_dir / "README.md"
    if not readme_path.exists():
        readme_content = _build_model_readme(train_args, label_files)
        readme_path.write_text(readme_content, encoding="utf-8")
        print(f"✓ README.md creado en {output_dir}")
    
    print(f"\n✓ Checkpoint verificado y listo para HF:")
    print(f"  📁 Ubicación: {output_dir}")
    print(f"  📦 Archivos: {len(list(output_dir.glob('*')))}")
    print(f"  🏷️ Labels: {', '.join(label_files)}")
    print(f"  ✅ Auto-contenido: Sí (puede usarse sin dependencias externas)")
    
    return output_dir


def upload_to_hf(
    repo_name: str,
    checkpoint_dir: str,
    token: str = None,
    private: bool = False,
    commit_message: str = "Upload PIBot Joint BERT model with full reproducibility"
):
    """Subir el checkpoint a Hugging Face Hub."""
    
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("❌ Error: huggingface-hub no está instalado")
        print("Instalar con: pip install huggingface-hub")
        return False
    
    checkpoint_dir = Path(checkpoint_dir)
    
    try:
        api = HfApi()
        
        # Crear repo si no existe
        print(f"\n🔄 Creando/verificando repositorio: {repo_name}")
        repo_url = create_repo(
            repo_name,
            repo_type="model",
            exist_ok=True,
            private=private,
            token=token
        )
        print(f"✓ Repositorio: {repo_url}")
        
        # Subir archivos
        print(f"\n📤 Subiendo checkpoint ({checkpoint_dir})...")
        api.upload_folder(
            folder_path=str(checkpoint_dir),
            repo_id=repo_name,
            repo_type="model",
            commit_message=commit_message,
            token=token,
        )
        
        print(f"\n✅ Modelo subido exitosamente a: {repo_url}")
        print(f"\n📝 Instrucciones de uso:")
        print(f"```python")
        print(f"from load_local_model import PIBotPredictor")
        print(f"predictor = PIBotPredictor('{repo_name}')")
        print(f"result = predictor.predict('tu texto aquí')")
        print(f"```")
        
        return True
        
    except Exception as e:
        print(f"❌ Error al subir: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Subir PIBot Joint BERT a HF de forma reproducible"
    )
    parser.add_argument(
        "--repo_name",
        required=True,
        help="Nombre del repositorio en HF (ej: usuario/pibot-jointbert)"
    )
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Directorio del modelo local"
    )
    parser.add_argument(
        "--label_dir",
        help="Directorio con *_label.txt (opcional, por defecto usa data/{task})"
    )
    parser.add_argument(
        "--code_dir",
        help="Directorio con modeling_jointbert.py/module.py/__init__.py (opcional, por defecto usa model_in/)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Crear repositorio privado"
    )
    parser.add_argument(
        "--token",
        help="Token de HF (opcional)"
    )
    parser.add_argument(
        "--commit_message",
        default="Upload PIBot Joint BERT model with full reproducibility",
        help="Mensaje del commit"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PIBot Joint BERT - Upload to Hugging Face")
    print("=" * 60)
    
    # Preparar checkpoint
    checkpoint_dir = prepare_checkpoint_for_hf(
        args.model_dir,
        label_dir=args.label_dir,
        code_dir=args.code_dir,
    )
    
    # Subir a HF
    success = upload_to_hf(
        repo_name=args.repo_name,
        checkpoint_dir=checkpoint_dir,
        token=args.token,
        private=args.private,
        commit_message=args.commit_message
    )
    
    if not success:
        exit(1)


if __name__ == "__main__":
    main()
