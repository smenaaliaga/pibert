"""
Prepara model_package/ copiando artefactos desde model_out, data y model_in.

Uso:
    python util/prepare_model_package.py --model_dir model_out/pibert_beto --labels_dir data/pibimacecv3

    # Limpiar model_package antes de copiar:
    python util/prepare_model_package.py --model_dir model_out/pibert_beto --labels_dir data/pibimacecv3 --clean

    # Preparar + subir a HF:
    python util/prepare_model_package.py --model_dir model_out/pibert_beto --labels_dir data/pibimacecv3 --clean --upload --repo_name smenaaliaga/pibert
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

# ── Rutas por defecto (relativas a la raíz del proyecto) ───────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CODE_DIR = PROJECT_ROOT / "model_in"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "model_package"

# ── Archivos esperados ──────────────────────────────────────────────

MODEL_FILES_REQUIRED = ["model.safetensors", "config.json"]

MODEL_FILES_OPTIONAL = [
    "training_args.bin",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "added_tokens.json",
    "spm.model",  # Para DeBERTa SentencePiece
]

CODE_FILES = [
    "modeling_jointbert.py",
    "module.py",
    "__init__.py",
]

LABEL_FILES = [
    "calc_mode_label.txt",
    "activity_label.txt",
    "region_label.txt",
    "investment_label.txt",
    "req_form_label.txt",
    "slot_label.txt",
]


# ── Utilidades ──────────────────────────────────────────────────────

def log_ok(msg: str):
    print(f"  ✅ {msg}")


def log_skip(msg: str):
    print(f"  ⏭️  {msg}")


def log_error(msg: str):
    print(f"  ❌ {msg}")


def detect_model_type(config_path: Path) -> str:
    """Detecta si es DeBERTa, BERT, etc, leyendo config.json"""
    if not config_path.exists():
        return "unknown"
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        model_name = config.get("_name_or_path", "").lower()
        if "mdeberta" in model_name or "deberta-v3" in model_name:
            return "deberta"
        return "bert"
    except Exception:
        return "unknown"


def copy_file(src: Path, dst: Path, required: bool = True) -> bool:
    """Copia un archivo. Retorna True si se copió sin error."""
    if src.exists():
        shutil.copy2(src, dst)
        log_ok(src.name)
        return True
    elif required:
        log_error(f"No encontrado (requerido): {src}")
        return False
    else:
        log_skip(f"No encontrado (opcional): {src.name}")
        return True


# ── Generación de README ────────────────────────────────────────────

def generate_readme(output_dir: Path, repo_name: str | None = None) -> bool:
    """
    Genera README.md (Model Card) a partir de config.json y labels.
    Retorna True si se generó correctamente.
    """
    config_path = output_dir / "config.json"
    if not config_path.exists():
        log_error("No se puede generar README: falta config.json")
        return False

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    base_model = config.get("_name_or_path", "bert-base-uncased")
    task = config.get("finetuning_task", "unknown")

    # Leer labels
    labels_dir = output_dir / "labels"
    intent_heads = {}
    slot_labels = []
    for label_file in sorted(labels_dir.glob("*_label.txt")):
        name = label_file.stem  # e.g. "calc_mode_label"
        head_name = name.replace("_label", "")  # e.g. "calc_mode"
        values = label_file.read_text(encoding="utf-8").strip().splitlines()
        if head_name == "slot":
            slot_labels = values
        else:
            intent_heads[head_name] = values

    # Determinar repo display name
    repo_display = repo_name or "smenaaliaga/pibert"

    # Construir tabla de intent heads
    intent_table = ""
    for head, values in intent_heads.items():
        vals = ", ".join(f"`{v}`" for v in values)
        intent_table += f"| `{head}` | {len(values)} | {vals} |\n"

    # Slot entities (solo B- tags)
    entities = sorted(set(
        tag.replace("B-", "").lower()
        for tag in slot_labels if tag.startswith("B-")
    ))
    entities_list = ", ".join(f"`{e}`" for e in entities)

    readme = f"""---
language: es
tags:
- intent-classification
- slot-filling
- joint-bert
- spanish
- economics
- chile
- multi-head
license: mit
base_model: {base_model}
pipeline_tag: token-classification
---

# PIBot Joint BERT

Modelo **Joint BERT multi-head** para clasificación de intención y slot filling,
especializado en consultas sobre indicadores macroeconómicos del Banco Central de Chile.

## Arquitectura

| Componente | Detalle |
|---|---|
| Base | `{base_model}` |
| Task | `{task}` |
| Intent heads | {len(intent_heads)} ({', '.join(f'`{h}`' for h in intent_heads)}) |
| Slot labels | {len(slot_labels)} (BIO) |
| Custom code | `modeling_jointbert.py`, `module.py` |

### Intent Heads

| Head | Clases | Valores |
|---|---|---|
{intent_table}
### Slot Entities (BIO)

Entidades extraídas: {entities_list}

Esquema BIO completo: {len(slot_labels)} etiquetas (`O`, `B-*`, `I-*`).

## Uso

### Instalación

```bash
pip install torch transformers
```

### Carga del Modelo

```python
import torch
from transformers import AutoTokenizer, AutoConfig

# Cargar tokenizer y config
tokenizer = AutoTokenizer.from_pretrained("{repo_display}", trust_remote_code=True)
config = AutoConfig.from_pretrained("{repo_display}", trust_remote_code=True)

# Cargar labels desde el repo
from huggingface_hub import hf_hub_download
import os

label_dir = os.path.dirname(hf_hub_download("{repo_display}", "labels/slot_label.txt"))

# Leer intent y slot labels
def read_labels(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]

slot_labels = read_labels(os.path.join(label_dir, "slot_label.txt"))

# Preparar intent_label_lst para cada head
intent_label_lst = []
for head in {list(intent_heads.keys())}:
    intent_label_lst.append(read_labels(os.path.join(label_dir, f"{{head}}_label.txt")))

# Cargar modelo con custom code
from transformers import AutoModelForTokenClassification
from modeling_jointbert import JointBERT  # auto-cargado con trust_remote_code

model = JointBERT.from_pretrained(
    "{repo_display}",
    config=config,
    intent_label_lst=intent_label_lst,
    slot_label_lst=slot_labels,
    trust_remote_code=True,
)
model.eval()
```

### Predicción

```python
text = "cuál fue el imacec de agosto 2024"
tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    outputs = model(**tokens)
    # outputs contiene intent_logits (lista) y slot_logits
```

## Estructura del Paquete

```
model_package/
├── config.json              # Configuración BERT + task
├── model.safetensors        # Pesos del modelo
├── tokenizer.json           # Tokenizer
├── tokenizer_config.json
├── special_tokens_map.json
├── vocab.txt
├── modeling_jointbert.py    # Arquitectura JointBERT (custom)
├── module.py                # CRF y módulos auxiliares
├── __init__.py
├── README.md                # Este archivo
└── labels/
    ├── slot_label.txt
{chr(10).join(f'    ├── {h}_label.txt' for h in intent_heads)}
```

## Datos de Entrenamiento

Entrenado con datos de consultas sobre indicadores macroeconómicos chilenos:
- **IMACEC** (Indicador Mensual de Actividad Económica)
- **PIB** (Producto Interno Bruto)
- Sectores económicos, frecuencias, períodos, regiones

## Limitaciones

- Especializado en consultas macroeconómicas del Banco Central de Chile
- Mejor rendimiento en consultas cortas (< 50 tokens)
- Requiere `trust_remote_code=True` por la arquitectura custom

## Cita

```bibtex
@misc{{pibot-jointbert,
  author = {{Banco Central de Chile}},
  title = {{PIBot Joint BERT - Multi-head Intent + Slot Filling}},
  year = {{2025}},
  publisher = {{Hugging Face}},
  howpublished = {{\\url{{https://huggingface.co/{repo_display}}}}}
}}
```

## Referencias

- [BERT for Joint Intent Classification and Slot Filling](https://arxiv.org/abs/1902.10909)
- [JointBERT implementation](https://github.com/monologg/JointBERT)
- [BETO: Spanish BERT](https://github.com/dccuchile/beto)

## Licencia

MIT License
"""

    readme_path = output_dir / "README.md"
    readme_path.write_text(readme.strip() + "\n", encoding="utf-8")
    log_ok("README.md generado")
    return True


# ── Validación ──────────────────────────────────────────────────────

def validate_inputs(model_dir: Path, labels_dir: Path, code_dir: Path) -> bool:
    ok = True
    for name, path in [("model_dir", model_dir), ("labels_dir", labels_dir), ("code_dir", code_dir)]:
        if not path.is_dir():
            log_error(f"No existe {name}: {path}")
            ok = False
        else:
            log_ok(f"{name}: {path}")
    return ok


# ── Lógica principal ───────────────────────────────────────────────

def prepare_package(
    model_dir: Path,
    labels_dir: Path,
    code_dir: Path,
    output_dir: Path,
    clean: bool = False,
    repo_name: str | None = None,
) -> bool:
    """
    Copia todos los artefactos necesarios a output_dir.
    Retorna True si todo salió bien.
    """

    print()
    print("=" * 60)
    print("  Preparar model_package")
    print("=" * 60)

    # ── Validar entradas ──
    print("\n📂 Validando directorios de entrada...")
    if not validate_inputs(model_dir, labels_dir, code_dir):
        return False

    # ── Limpiar si se solicita ──
    if clean and output_dir.exists():
        print(f"\n🧹 Limpiando {output_dir}...")
        shutil.rmtree(output_dir)
        log_ok(f"Eliminado {output_dir}")

    # ── Crear estructura ──
    labels_out = output_dir / "labels"
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    errors = False

    # ── 1) Pesos y config (requeridos) ──
    print("\n📦 [1/5] Pesos y configuración...")
    for fname in MODEL_FILES_REQUIRED:
        if not copy_file(model_dir / fname, output_dir / fname, required=True):
            errors = True

    # ── 2) Tokenizer y archivos opcionales ──
    print("\n📝 [2/5] Tokenizer y archivos opcionales...")
    for fname in MODEL_FILES_OPTIONAL:
        src = model_dir / fname
        # vocab.txt: fallback a labels_dir si no está en model_dir
        if fname == "vocab.txt" and not src.exists():
            src = labels_dir / fname
        copy_file(src, output_dir / fname, required=False)
    
    # ── Regenerar tokenizer.json si falta y es DeBERTa ──
    model_type = detect_model_type(model_dir / "config.json")
    if model_type == "deberta" and not (output_dir / "tokenizer.json").exists():
        print("\n   📋 DeBERTa: regenerando tokenizer.json...")
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=False)
            tokenizer.save_pretrained(str(output_dir))
            log_ok("tokenizer.json regenerado desde modelo")
        except Exception as e:
            log_skip(f"No se pudo regenerar tokenizer.json ({type(e).__name__})")
            if not (output_dir / "vocab.txt").exists():
                log_error("No hay vocab.txt. DeBERTa requiere al menos vocab.txt")

    # ── 3) Código custom ──
    print("\n🐍 [3/5] Código custom (desde model_in/)...")
    for fname in CODE_FILES:
        if not copy_file(code_dir / fname, output_dir / fname, required=True):
            errors = True

    # ── 4) Labels ──
    print("\n🏷️  [4/5] Labels...")
    for fname in LABEL_FILES:
        if not copy_file(labels_dir / fname, labels_out / fname, required=True):
            errors = True

    # ── 5) README.md ──
    print("\n📄 [5/5] README.md (Model Card)...")
    if not generate_readme(output_dir, repo_name=repo_name):
        errors = True

    # ── Resumen ──
    print()
    if errors:
        print("⚠️  Preparación completada CON ERRORES. Revisa los mensajes arriba.")
        return False
    else:
        files = [f for f in sorted(output_dir.rglob("*")) if f.is_file()]
        print(f"✅ model_package listo: {output_dir.resolve()}")
        print(f"   Archivos totales: {len(files)}")
        print()
        print("   Estructura:")
        for f in files:
            rel = f.relative_to(output_dir)
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"   ├── {rel}  ({size_mb:.1f} MB)")
        print()
        return True


# ── CLI ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepara model_package/ copiando artefactos del modelo entrenado.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python util/prepare_model_package.py --model_dir model_out/pibot_v5 --labels_dir data/pibimacecv5
  python util/prepare_model_package.py --model_dir model_out/pibot_v5 --labels_dir data/pibimacecv5 --clean
  python util/prepare_model_package.py --model_dir model_out/pibot_v5 --labels_dir data/pibimacecv5 --clean --upload --repo_name user/model
        """,
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Ruta al modelo entrenado (ej: model_out/pibot_model_v5_beto_crf)",
    )
    parser.add_argument(
        "--labels_dir",
        type=str,
        required=True,
        help="Ruta al dataset con los *_label.txt (ej: data/pibimacecv5)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Eliminar model_package/ antes de copiar (evita mezclar versiones)",
    )

    # Opciones de upload
    upload_group = parser.add_argument_group("Upload a Hugging Face (opcional)")
    upload_group.add_argument(
        "--upload",
        action="store_true",
        help="Subir a HF después de preparar",
    )
    upload_group.add_argument(
        "--repo_name",
        type=str,
        default=None,
        help="Nombre del repo en HF (ej: smenaaliaga/pibert). Requerido si --upload",
    )
    upload_group.add_argument(
        "--private",
        action="store_true",
        help="Crear repo como privado en HF",
    )

    args = parser.parse_args()

    if args.upload and not args.repo_name:
        parser.error("--repo_name es requerido cuando se usa --upload")

    return args


def main():
    args = parse_args()

    # Resolver rutas relativas desde la raíz del proyecto
    model_dir = Path(args.model_dir)
    labels_dir = Path(args.labels_dir)

    if not model_dir.is_absolute():
        model_dir = PROJECT_ROOT / model_dir
    if not labels_dir.is_absolute():
        labels_dir = PROJECT_ROOT / labels_dir

    # Preparar
    success = prepare_package(
        model_dir=model_dir,
        labels_dir=labels_dir,
        code_dir=DEFAULT_CODE_DIR,
        output_dir=DEFAULT_OUTPUT_DIR,
        clean=args.clean,
        repo_name=getattr(args, 'repo_name', None),
    )

    if not success:
        sys.exit(1)

    # Upload opcional
    if args.upload:
        print("=" * 60)
        print("  Subiendo a Hugging Face...")
        print("=" * 60)

        upload_script = Path(__file__).resolve().parent / "upload_model_package.py"

        if not upload_script.exists():
            log_error(f"No se encontró {upload_script}")
            sys.exit(1)

        cmd = [
            sys.executable,
            str(upload_script),
            "--repo_name", args.repo_name,
            "--model_package_dir", str(DEFAULT_OUTPUT_DIR),
        ]
        if args.private:
            cmd.append("--private")

        result = subprocess.run(cmd)
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
