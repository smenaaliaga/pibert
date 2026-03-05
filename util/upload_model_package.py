"""
Sube model_package/ a Hugging Face Hub.

El directorio model_package/ ya es auto-contenido con:
- Pesos (model.safetensors)
- Config (config.json)
- Tokenizer (tokenizer.json, vocab.txt, etc.)
- Código custom (modeling_jointbert.py, module.py, __init__.py)
- Labels (labels/*.txt)
- Training args (training_args.bin)
- README.md (Model Card)

Uso:
    python upload_model_package.py --repo_name tu-usuario/pibot-jointbert
    python upload_model_package.py --repo_name tu-usuario/pibot-jointbert --private
    python upload_model_package.py --repo_name tu-usuario/pibot-jointbert --token hf_xxx --private
"""

import argparse
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("❌ huggingface-hub no está instalado. Instalar con: pip install huggingface-hub")
    sys.exit(1)


MODEL_PACKAGE_DIR = Path(__file__).resolve().parents[1] / "model_package"

# Archivos que deben existir en model_package para ser un paquete válido
REQUIRED_FILES = [
    "config.json",
    "modeling_jointbert.py",
    "module.py",
    "__init__.py",
    "tokenizer_config.json",
    "README.md",
]

REQUIRED_WEIGHT_FILES = ["model.safetensors", "pytorch_model.bin"]

REQUIRED_LABEL_FILES = [
    "labels/calc_mode_label.txt",
    "labels/activity_label.txt",
    "labels/region_label.txt",
    "labels/investment_label.txt",
    "labels/req_form_label.txt",
    "labels/slot_label.txt",
]


def detect_model_type(package_dir: Path) -> str:
    """Detecta si es DeBERTa, BERT, etc, leyendo config.json"""
    config_path = package_dir / "config.json"
    if not config_path.exists():
        return "unknown"
    
    import json
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        model_name = config.get("_name_or_path", "").lower()
        if "mdeberta" in model_name or "deberta-v3" in model_name:
            return "deberta"
        return "bert"
    except Exception:
        return "unknown"


def validate_model_package(package_dir: Path) -> bool:
    """Valida que model_package tenga todos los archivos necesarios."""
    ok = True

    if not package_dir.exists():
        print(f"❌ No se encuentra el directorio: {package_dir}")
        return False

    # Detectar tipo de modelo
    model_type = detect_model_type(package_dir)
    print(f"   Tipo de modelo detectado: {model_type}")

    # Pesos del modelo
    has_weights = any((package_dir / f).exists() for f in REQUIRED_WEIGHT_FILES)
    if not has_weights:
        print(f"❌ Falta archivo de pesos: se requiere alguno de {REQUIRED_WEIGHT_FILES}")
        ok = False

    # Archivos core
    for f in REQUIRED_FILES:
        if not (package_dir / f).exists():
            print(f"❌ Falta archivo requerido: {f}")
            ok = False

    # Labels
    missing_labels = [f for f in REQUIRED_LABEL_FILES if not (package_dir / f).exists()]
    if missing_labels:
        print(f"⚠ Labels faltantes: {missing_labels}")
        ok = False

    # Tokenizer: validación flexible por tipo de modelo
    has_tokenizer_json = (package_dir / "tokenizer.json").exists()
    has_vocab = (package_dir / "vocab.txt").exists()
    has_spm_model = (package_dir / "spm.model").exists()
    has_tokenizer_config = (package_dir / "tokenizer_config.json").exists()
    
    if model_type == "deberta":
        # DeBERTa slow: necesita (spm.model O vocab.txt) + tokenizer_config.json
        # spm.model es para SentencePiece, vocab.txt es fallback
        has_vocab_equivalent = has_vocab or has_spm_model
        if not has_vocab_equivalent or not has_tokenizer_config:
            print(f"⚠ DeBERTa: spm.model={has_spm_model}, vocab.txt={has_vocab}, tokenizer_config.json={has_tokenizer_config}")
            if not has_vocab_equivalent:
                print("❌ DeBERTa requiere spm.model o vocab.txt")
                ok = False
            if not has_tokenizer_config:
                print("❌ DeBERTa requiere tokenizer_config.json")
                ok = False
        else:
            print(f"   ✓ DeBERTa valid: spm.model={has_spm_model}, tokenizer.json={has_tokenizer_json}")
    else:
        # BERT/otros: tokenizer.json es preferible, vocab.txt es fallback
        if not has_tokenizer_json and not has_vocab:
            print("❌ Falta tokenizer: se requiere tokenizer.json o vocab.txt")
            ok = False
        elif not has_tokenizer_json:
            print("⚠ Falta tokenizer.json (pero vocab.txt existe, puede funcionar)")

    return ok


def upload(
    repo_name: str,
    package_dir: Path,
    token: str = None,
    private: bool = False,
    commit_message: str = "Upload PIBot Joint BERT model package",
):
    """Sube model_package/ a Hugging Face Hub."""

    api = HfApi(token=token)

    # Verificar si el repo ya existe (no requiere permisos de creación)
    print(f"\n📦 Verificando repositorio: {repo_name}")
    try:
        api.repo_info(repo_id=repo_name, repo_type="model", token=token)
        print("✓ Repositorio existente encontrado")
    except Exception:
        print("ℹ️ Repositorio no encontrado. Intentando crearlo...")
        try:
            repo_url = create_repo(
                repo_id=repo_name,
                token=token,
                private=private,
                exist_ok=True,
                repo_type="model",
            )
            print(f"✓ Repositorio creado: {repo_url}")
        except Exception as create_error:
            raise RuntimeError(
                "No se pudo crear el repositorio en Hugging Face. "
                "Si usas un namespace de organización (ej: org/repo), tu cuenta/token necesita permisos "
                "para crear repositorios en esa organización. "
                "Alternativas: (1) pide a un admin de la org que cree el repo y vuelve a ejecutar, "
                "o (2) sube a tu namespace personal (ej: tu-usuario/nombre-repo). "
                f"Detalle: {create_error}"
            ) from create_error

    # Listar archivos que se subirán
    all_files = sorted(package_dir.rglob("*"))
    files_to_upload = [f for f in all_files if f.is_file() and f.name != ".gitkeep"]
    print(f"\n📤 Subiendo {len(files_to_upload)} archivos desde {package_dir}...")
    for f in files_to_upload:
        print(f"   • {f.relative_to(package_dir)}")

    # Subir
    try:
        url = api.upload_folder(
            folder_path=str(package_dir),
            repo_id=repo_name,
            repo_type="model",
            commit_message=commit_message,
            token=token,
            ignore_patterns=[".gitkeep", "__pycache__/**"],
        )
    except Exception as upload_error:
        raise RuntimeError(
            "No se pudo subir al repositorio. Verifica que tu token tenga permiso de escritura "
            "sobre ese repo (sobre todo en organizaciones) y que el repo_id sea correcto. "
            f"Detalle: {upload_error}"
        ) from upload_error

    print(f"\n{'='*60}")
    print(f"✅ MODELO SUBIDO EXITOSAMENTE")
    print(f"{'='*60}")
    print(f"\n🔗 https://huggingface.co/{repo_name}")
    print(f"📦 Commit: {url}")
    print(f"\n📝 Para usar en otro proyecto:")
    print(f"   from huggingface_hub import snapshot_download")
    print(f"   model_dir = snapshot_download('{repo_name}')")
    print(f"\n   O directamente con transformers:")
    print(f"   from transformers import AutoModel, AutoTokenizer")
    print(f"   tokenizer = AutoTokenizer.from_pretrained('{repo_name}', trust_remote_code=True)")
    print(f"   model = AutoModel.from_pretrained('{repo_name}', trust_remote_code=True, ...)")


def main():
    parser = argparse.ArgumentParser(
        description="Subir model_package/ a Hugging Face Hub"
    )
    parser.add_argument(
        "--repo_name",
        required=True,
        help="Nombre del repositorio en HF (ej: tu-usuario/pibot-jointbert)",
    )
    parser.add_argument(
        "--model_package_dir",
        default=str(MODEL_PACKAGE_DIR),
        help=f"Directorio del paquete del modelo (default: {MODEL_PACKAGE_DIR})",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Crear repositorio privado",
    )
    parser.add_argument(
        "--token",
        help="Token de Hugging Face (opcional, usa el guardado con `huggingface-cli login`)",
    )
    parser.add_argument(
        "--commit_message",
        default="Upload PIBot Joint BERT model package",
        help="Mensaje del commit",
    )

    args = parser.parse_args()
    package_dir = Path(args.model_package_dir)

    print("=" * 60)
    print("PIBot Joint BERT — Upload model_package to Hugging Face")
    print("=" * 60)

    # Validar
    print(f"\n🔍 Validando {package_dir}...")
    if not validate_model_package(package_dir):
        print("\n❌ Validación fallida. Corrige los errores antes de subir.")
        sys.exit(1)
    print("✓ Validación OK\n")

    # Subir
    try:
        upload(
            repo_name=args.repo_name,
            package_dir=package_dir,
            token=args.token,
            private=args.private,
            commit_message=args.commit_message,
        )
    except Exception as e:
        print(f"\n❌ Error al subir: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
