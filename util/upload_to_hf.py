"""
Script para subir el modelo PIBot a Hugging Face Hub.

Sube el checkpoint entrenado y los archivos de soporte (tokenizer, código custom, labels) para que otros proyectos puedan usarlo con trust_remote_code.

Requisitos:
    pip install huggingface-hub
    
Uso básico:
    python util/upload_to_hf.py --repo_name <usuario>/<nombre-repo> --model_dir <directorio-modelo> --label_dir <directorio-labels> 

Ejemplo:
    python util/upload_to_hf.py --repo_name smenaaliaga/pibot-jointbert --model_dir model_out/pibot_model_v3 --label_dir data/pibimacecv3
"""

import argparse
import shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder


def upload_model_to_hf(
    repo_name: str,
    model_dir: str,
    label_dir: str,
    code_dir: str = None,
    token: str = None,
    private: bool = False,
    commit_message: str = "Upload PIBot Joint BERT model"
):
    """
    Sube el modelo a Hugging Face Hub.
    
    Args:
        repo_name: Nombre del repositorio en HF (ej: 'tu-usuario/pibot-jointbert-beto')
        model_dir: Directorio local del modelo (ej: 'model_out/pibot_model_beto')
        label_dir: Ruta a la carpeta con los *_label.txt (ej: 'data/pibimacecv3')
        code_dir: Carpeta con modeling_jointbert.py/module.py/__init__.py (se usa si faltan en model_dir)
        token: Token de Hugging Face (opcional, usa el guardado si no se provee)
        private: Si True, crea un repositorio privado
        commit_message: Mensaje del commit
    """
    try:
        print("✓ huggingface-hub importado correctamente")
    except ImportError:
        print("❌ Error: huggingface-hub no está instalado")
        print("Instalar con: pip install huggingface-hub")
        return False
    
    # Verificar que el directorio del modelo existe
    model_path = Path(model_dir)
    if not model_path.exists():
        print(f"❌ Error: No se encuentra el directorio del modelo: {model_dir}")
        return False

    # Verificar archivos esenciales del modelo/tokenizer/código/labels
    core_weight_files = ['model.safetensors', 'pytorch_model.bin']
    core_config_file = 'config.json'
    tokenizer_files = ['tokenizer.json', 'tokenizer_config.json', 'vocab.txt', 'special_tokens_map.json']
    code_files = ['modeling_jointbert.py', 'module.py', '__init__.py']
    label_files = [
        'indicator_label.txt', 'metric_type_label.txt', 'seasonal_label.txt',
        'activity_label.txt', 'frequency_label.txt', 'req_form_label.txt', 'calc_mode_label.txt'
    ]

    has_weight = any((model_path / f).exists() for f in core_weight_files)
    if not has_weight:
        print(f"❌ Error: No se encontró ninguno de los archivos de pesos: {core_weight_files}")
        return False

    missing_core = [core_config_file] if not (model_path / core_config_file).exists() else []
    missing_tokenizer = [f for f in tokenizer_files if not (model_path / f).exists()]
    missing_code = [f for f in code_files if not (model_path / f).exists()]
    missing_labels = [f for f in label_files if not (model_path / f).exists()]

    # Completar labels faltantes copiando desde label_dir
    if missing_labels and label_dir:
        src_labels = Path(label_dir)
        if not src_labels.exists():
            print(f"⚠ Advertencia: label_dir no existe: {label_dir}")
        else:
            for f in missing_labels:
                src = src_labels / f
                dst = model_path / f
                if src.exists():
                    shutil.copyfile(src, dst)
            missing_labels = [f for f in label_files if not (model_path / f).exists()]

    # Completar código faltante copiando desde code_dir
    if missing_code and code_dir:
        src_code = Path(code_dir)
        if not src_code.exists():
            print(f"⚠ Advertencia: code_dir no existe: {code_dir}")
        else:
            for f in missing_code:
                src = src_code / f
                dst = model_path / f
                if src.exists():
                    shutil.copyfile(src, dst)
            missing_code = [f for f in code_files if not (model_path / f).exists()]

    if missing_core or missing_tokenizer or missing_code or missing_labels:
        print("⚠ Advertencia: faltan archivos que mejor subas para compatibilidad completa en HF:")
        if missing_core:
            print(f"  - Core: {missing_core}")
        if missing_tokenizer:
            print(f"  - Tokenizer: {missing_tokenizer}")
        if missing_code:
            print(f"  - Código custom (trust_remote_code): {missing_code}")
        if missing_labels:
            print(f"  - Labels: {missing_labels}")
        print("  El modelo puede no cargar bien sin ellos (tokenizer, mapeos o código custom).")
    
    print(f"\n{'='*70}")
    print(f"SUBIENDO MODELO A HUGGING FACE HUB")
    print(f"{'='*70}\n")
    
    print(f"Directorio local: {model_path.absolute()}")
    print(f"Repositorio HF: {repo_name}")
    print(f"Privado: {private}\n")
    
    try:
        # Inicializar API
        api = HfApi(token=token)
        
        # Crear repositorio si no existe
        print("📦 Creando repositorio en Hugging Face...")
        try:
            create_repo(
                repo_id=repo_name,
                token=token,
                private=private,
                exist_ok=True,
                repo_type="model"
            )
            print(f"✓ Repositorio creado/verificado: {repo_name}\n")
        except Exception as e:
            print(f"⚠ Nota: {e}")
            print("  (Esto es normal si el repositorio ya existe)\n")
        
        # Subir archivos
        print("📤 Subiendo archivos del modelo...")
        url = upload_folder(
            folder_path=str(model_path),
            repo_id=repo_name,
            token=token,
            commit_message=commit_message,
            repo_type="model"
        )
        
        print(f"\n{'='*70}")
        print(f"✅ MODELO SUBIDO EXITOSAMENTE")
        print(f"{'='*70}\n")
        
        print(f"🔗 URL del modelo: https://huggingface.co/{repo_name}")
        print(f"📦 Commit: {url}\n")
        
        print("Para usar el modelo desde HF:")
        print(f"  from transformers import AutoModel")
        print(f"  model = AutoModel.from_pretrained('{repo_name}')\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error al subir el modelo: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_model_card(model_dir: str, output_path: str = None):
    """
    Crea un README.md (Model Card) para el repositorio de HF.
    
    Args:
        model_dir: Directorio del modelo
        output_path: Donde guardar el README (por defecto: model_dir/README.md)
    """
    if output_path is None:
        output_path = Path(model_dir) / "README.md"
    
    # Obtener información del modelo
    import torch
    args_path = Path(model_dir) / 'training_args.bin'
    
    if args_path.exists():
        args = torch.load(args_path, weights_only=False)  # Safe: loading our own training args
        task = args.task
        model_type = args.model_type
        epochs = args.num_train_epochs
    else:
        task = "pibimacec"
        model_type = "beto"
        epochs = "N/A"
    
    readme_content = f'''---
language: es
tags:
- multi-classification
- joint-bert
- spanish
- economics
- chile
license: mit
---

# PIBot Joint BERT - {model_type.upper()}

Modelo Joint BERT multi-cabeza entrenado para consultas económicas del Banco Central de Chile. Predice 7 cabezas de clasificación en paralelo: indicator, metric_type, seasonal, activity, frequency, req_form, calc_mode.

## Descripción del Modelo

- **Arquitectura base**: BERT (dccuchile/bert-base-spanish-wwm-cased)
- **Idioma**: Español
- **Task/dataset**: {task}
- **Épocas de entrenamiento**: {epochs}
- **Heads**: 7 clasificadores independientes (sin slots/CRF en esta versión)

## Uso

### Instalación

```bash
pip install torch transformers
```

### Ejemplo de carga

```python
import torch
from transformers import AutoTokenizer, AutoModel

repo_id = "<tu-usuario>/<tu-repo>"  # e.g., smenaaliaga/pibot_model_v3

tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)

text = "cual fue el pib del ultimo trimestre"
batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
with torch.no_grad():
    outputs = model(**batch)
    logits = outputs[1]  # tuple: (indicator, metric_type, calc_mode, seasonal, req_form, frequency, activity)

# Cada logits[i] -> argmax -> índice -> label usando los *_label.txt incluidos en el repo
```

### Archivos incluidos
- Pesos: model.safetensors (o pytorch_model.bin)
- Config: config.json
- Tokenizer: tokenizer.json, tokenizer_config.json, vocab.txt, special_tokens_map.json
- Código custom: modeling_jointbert.py, module.py, __init__.py (usar trust_remote_code=True)
- Labels: indicator_label.txt, metric_type_label.txt, seasonal_label.txt, activity_label.txt, frequency_label.txt, req_form_label.txt, calc_mode_label.txt
- Opcional: training_args.bin, README.md (esta model card)

## Datos de Entrenamiento (resumen)
- Consultas sobre IMACEC/PIB y variaciones (YoY, periodo previo), frecuencias (m/q/a), actividades (total, sectores), formas de solicitud (latest/point/range), estacionalidad (sa/nsa).

## Notas
- Se recomienda `trust_remote_code=True` para cargar el modelo custom.
- Las labels están en texto plano para mapear argmax→string.
- Ajustar `max_length` según el largo de consultas (64 recomendado).

## Licencia

MIT License

'''
    
    Path(output_path).write_text(readme_content, encoding='utf-8')
    print(f"✓ Model card creado: {output_path}")
    return output_path


def login_to_hf():
    """
    Ayuda al usuario a hacer login en Hugging Face.
    """
    try:
        from huggingface_hub import login, HfFolder
        
        print("\n" + "="*70)
        print("LOGIN EN HUGGING FACE")
        print("="*70 + "\n")
        
        print("Opciones para autenticación:\n")
        print("1. Usar token guardado (si ya hiciste login antes)")
        print("2. Login interactivo (abre navegador)")
        print("3. Proveer token manualmente\n")
        
        # Verificar si ya hay token guardado
        token = HfFolder.get_token()
        if token:
            print(f"✓ Token encontrado en cache")
            return token
        
        choice = input("Elige opción (1/2/3): ").strip()
        
        if choice == "2":
            print("\nAbriendo navegador para login...")
            login()
            print("✓ Login exitoso")
            return HfFolder.get_token()
        
        elif choice == "3":
            print("\nObtén tu token en: https://huggingface.co/settings/tokens")
            token = input("Pega tu token aquí: ").strip()
            login(token=token)
            print("✓ Token guardado")
            return token
        
        else:
            print("Usando token en cache...")
            return HfFolder.get_token()
            
    except ImportError:
        print("❌ huggingface-hub no instalado")
        print("Instalar con: pip install huggingface-hub")
        return None


def main():
    parser = argparse.ArgumentParser(description="Subir modelo PIBot a Hugging Face Hub")
    
    parser.add_argument(
        '--repo_name',
        type=str,
        required=True,
        help='Nombre del repo en HF (formato: usuario/nombre-modelo)'
    )
    
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help='Directorio del modelo a subir'
    )

    parser.add_argument(
        '--label_dir',
        type=str,
        required=True,
        help='Directorio donde están los *_label.txt; se copiarán si faltan en model_dir'
    )

    parser.add_argument(
        '--code_dir',
        type=str,
        default=None,
        help='Directorio con modeling_jointbert.py/module.py/__init__.py para completar si faltan'
    )
    
    parser.add_argument(
        '--token',
        type=str,
        default=None,
        help='Token de Hugging Face (opcional si ya hiciste login)'
    )
    
    parser.add_argument(
        '--private',
        action='store_true',
        help='Crear repositorio privado'
    )
    
    parser.add_argument(
        '--create_model_card',
        action='store_true',
        help='Crear README.md antes de subir'
    )

    parser.add_argument(
        '--commit_message',
        type=str,
        default='Upload PIBot Joint BERT model',
        help='Mensaje de commit para el push a HF'
    )
    
    parser.add_argument(
        '--no_login',
        action='store_true',
        help='No intentar hacer login (usar token existente)'
    )
    
    args = parser.parse_args()
    
    # Login si no hay token y no se especificó --no_login
    if args.token is None and not args.no_login:
        args.token = login_to_hf()
        if args.token is None:
            print("\n❌ No se pudo obtener token de autenticación")
            print("Usa --token YOUR_TOKEN o ejecuta primero: huggingface-cli login")
            return
    
    # Crear model card si se solicita
    if args.create_model_card:
        print("\n📝 Creando Model Card...")
        create_model_card(args.model_dir)
        print()
    
    # Subir modelo
    success = upload_model_to_hf(
        repo_name=args.repo_name,
        label_dir=args.label_dir,
        model_dir=args.model_dir,
        code_dir=args.code_dir,
        token=args.token,
        private=args.private,
        commit_message=args.commit_message
    )
    
    if success:
        print("\n🎉 ¡Proceso completado exitosamente!")
    else:
        print("\n❌ Hubo errores en el proceso")


if __name__ == '__main__':
    main()
