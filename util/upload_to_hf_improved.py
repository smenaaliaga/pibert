"""
Script mejorado para subir PIBot Joint BERT a Hugging Face Hub de forma reproducible.

Este script crea un checkpoint completamente auto-contenido que puede ser:
1. Cargado localmente con: PIBotPredictor("path")
2. Descargado y usado directamente desde HF
3. Reproducible en cualquier máquina sin dependencias externas

Uso:
    python upload_to_hf_improved.py \
        --repo_name username/pibot-jointbert \
        --model_dir model_out/pibot_model_v3 \
        --private
"""

import argparse
import json
import shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder


def prepare_checkpoint_for_hf(model_dir: str, output_dir: str = None):
    """
    Prepara el checkpoint para ser auto-contenido en HF.
    
    Asegura que contiene:
    - model.safetensors (pesos)
    - config.json (configuración)
    - tokenizer.json + tokenizer_config.json + vocab.txt (tokenizer)
    - training_args.bin (parámetros de entrenamiento)
    - *_label.txt (labels de cada cabeza)
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
    
    # Verificar archivos requeridos
    required_files = [
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "training_args.bin",
        "vocab.txt",
        "indicator_label.txt",
        "metric_type_label.txt",
        "calc_mode_label.txt",
        "seasonal_label.txt",
        "req_form_label.txt",
        "frequency_label.txt",
        "activity_label.txt",
    ]
    
    missing = [f for f in required_files if not (output_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Archivos faltantes en checkpoint: {missing}")
    
    # Crear README si no existe
    readme_path = output_dir / "README.md"
    if not readme_path.exists():
        readme_content = """# PIBot Joint BERT - 7 Heads

Modelo Joint BERT para clasificación multi-cabeza de consultas sobre indicadores económicos.

## Cabezas de Clasificación

El modelo predice simultáneamente 7 atributos:
- **indicator**: Indicador económico (ej: imacec, pib)
- **metric_type**: Tipo de métrica (ej: index, level)
- **calc_mode**: Modo de cálculo (ej: yoy, mom)
- **seasonal**: Ajuste estacional (ej: sa, nsa)
- **req_form**: Forma de solicitud (ej: latest, historical)
- **frequency**: Frecuencia (ej: m, q, a)
- **activity**: Actividad/Sector (ej: total, agriculture)

## Uso

### Opción 1: Local (Recomendado para máxima compatibilidad)

```python
from load_local_model import PIBotPredictor

predictor = PIBotPredictor("path/to/model")
result = predictor.predict("cual fue el pib del último trimestre")
print(result)
```

### Opción 2: Desde Hugging Face Hub

```python
from load_local_model import PIBotPredictor

# Descargar y usar
predictor = PIBotPredictor("username/pibot-jointbert")
result = predictor.predict("cual fue el imacec")
print(result)
```

### Línea de comandos

```bash
python load_local_model.py --model_dir path/to/model --text "tu consulta"
```

## Estructura del Checkpoint

```
model_dir/
├── model.safetensors              # Pesos del modelo
├── config.json                    # Configuración de BERT
├── training_args.bin              # Argumentos de entrenamiento
├── tokenizer.json                 # Tokenizer rápido
├── tokenizer_config.json
├── vocab.txt
├── modeling_jointbert.py          # Arquitectura custom
├── module.py                      # Clasificadores custom
├── __init__.py
├── *_label.txt                    # Labels para cada cabeza (7 archivos)
└── README.md
```

## Detalles Técnicos

- **Base Model**: dccuchile/bert-base-spanish-wwm-cased (BETO)
- **Framework**: PyTorch + Transformers
- **Formato de pesos**: SafeTensors
- **Tokenizer**: AutoTokenizer con use_fast=True

## Licencia

[Especificar licencia]

## Autor

[Tu información]
"""
        readme_path.write_text(readme_content, encoding="utf-8")
        print(f"✓ README.md creado en {output_dir}")
    
    # Verificar que los archivos de código estén presentes
    code_files = ["modeling_jointbert.py", "module.py", "__init__.py"]
    for cf in code_files:
        if not (output_dir / cf).exists():
            raise FileNotFoundError(f"Código faltante: {cf}")
    
    print(f"\n✓ Checkpoint verificado y listo para HF:")
    print(f"  📁 Ubicación: {output_dir}")
    print(f"  📦 Archivos: {len(list(output_dir.glob('*')))}")
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
    checkpoint_dir = prepare_checkpoint_for_hf(args.model_dir)
    
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
