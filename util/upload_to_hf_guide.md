# Publicar model_package en Hugging Face Hub

Guía paso a paso para preparar `model_package/` y subirlo a Hugging Face usando `upload_model_package.py`.

---

## 1) Prerrequisitos

- Python 3.10+ con entorno virtual activo
- `huggingface-hub` instalado:

```bash
pip install huggingface-hub
```

- Cuenta en [huggingface.co](https://huggingface.co) con un token de escritura (Settings → Access Tokens → `write`)

---

## 2) Login en Hugging Face (una sola vez)

```bash
huggingface-cli login
```

Pegar el token cuando se solicite. Queda guardado en `~/.cache/huggingface/token`.

---

## 3) Preparar model_package

El directorio `model_package/` debe ser **auto-contenido**: pesos, tokenizer, código custom, labels y README.

### Estructura esperada

```text
model_package/
├── config.json                 # Configuración del modelo
├── model.safetensors           # Pesos del modelo (o pytorch_model.bin)
├── training_args.bin           # Argumentos de entrenamiento
├── tokenizer.json              # Tokenizer
├── tokenizer_config.json
├── vocab.txt
├── special_tokens_map.json
├── modeling_jointbert.py       # Código custom del modelo
├── module.py                   # Clasificadores por cabeza
├── __init__.py                 # Exports para trust_remote_code
├── README.md                   # Model Card para HF
└── labels/
    ├── calc_mode_label.txt
    ├── activity_label.txt
    ├── region_label.txt
    ├── investment_label.txt
    ├── req_form_label.txt
    └── slot_label.txt
```

### Copiar archivos desde un modelo entrenado

Si entrenaste un modelo nuevo y necesitas actualizar `model_package/`, usa el script `prepare_model_package.py`:

```bash
# Básico: copia artefactos a model_package/
python util/prepare_model_package.py \
  --model_dir model_out/pibot_model_v5_beto_crf \
  --labels_dir data/pibimacecv5

# Con limpieza previa (recomendado al cambiar de modelo):
python util/prepare_model_package.py \
  --model_dir model_out/pibot_model_v5_beto_crf \
  --labels_dir data/pibimacecv5 \
  --clean

# Preparar + subir a HF en un solo paso:
python util/prepare_model_package.py \
  --model_dir model_out/pibot_model_v5_beto_crf \
  --labels_dir data/pibimacecv5 \
  --clean \
  --upload --repo_name smenaaliaga/pibert
```

El script:

1. Valida que `model_dir` y `labels_dir` existan
2. Si `--clean`, elimina `model_package/` previo (evita mezclar versiones)
3. Copia pesos (`model.safetensors`, `config.json`) desde `model_dir`
4. Copia tokenizer y archivos opcionales desde `model_dir`
5. Copia código custom (`modeling_jointbert.py`, `module.py`, `__init__.py`) desde `model_in/`
6. Copia los `*_label.txt` a `model_package/labels/`
7. Si `--upload`, llama a `upload_model_package.py` automáticamente

| Argumento | Descripción | Default |
|---|---|---|
| `--model_dir` | Ruta al modelo entrenado | **requerido** |
| `--labels_dir` | Ruta al dataset con los `*_label.txt` | **requerido** |
| `--clean` | Eliminar `model_package/` antes de copiar | `False` |
| `--upload` | Subir a HF después de preparar | `False` |
| `--repo_name` | Repo en HF (requerido si `--upload`) | — |
| `--private` | Crear repo privado en HF | `False` |

> **Tip**: Solo necesitas ejecutar este script cuando entrenas un modelo diferente. Si `model_package/` ya está completo, puedes ir directo al paso 4.

---

## 4) Subir a Hugging Face

Ejecutar desde la raíz del proyecto:

```bash
# Repo público
python util/upload_model_package.py --repo_name tu-usuario/pibert

# Repo privado
python util/upload_model_package.py --repo_name tu-usuario/pibert --private

# Con token explícito (si no hiciste huggingface-cli login)
python util/upload_model_package.py --repo_name tu-usuario/pibert --token hf_xxxxx --private
```

El script:

1. **Valida** que `model_package/` tenga todos los archivos requeridos
2. **Crea** el repositorio en HF si no existe
3. **Sube** todos los archivos (ignora `.gitkeep` y `__pycache__/`)
4. Muestra la **URL** del repositorio y ejemplos de uso

### Argumentos disponibles

| Argumento | Descripción | Default |
|---|---|---|
| `--repo_name` | Nombre del repo en HF (ej: `usuario/pibot-jointbert`) | **requerido** |
| `--model_package_dir` | Ruta al directorio del paquete | `model_package/` |
| `--private` | Crear como repositorio privado | `False` |
| `--token` | Token de HF (usa el del login si se omite) | `None` |
| `--commit_message` | Mensaje del commit | `"Upload PIBot Joint BERT model package"` |

---

## 5) Usar el modelo desde otro proyecto

Una vez subido, el modelo se puede consumir desde cualquier proyecto:

### Descargar snapshot completo

```python
from huggingface_hub import snapshot_download

model_dir = snapshot_download("tu-usuario/pibot-jointbert")
# model_dir contiene todos los archivos del model_package
```

### Cargar con transformers (trust_remote_code)

```python
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModel
from huggingface_hub import hf_hub_download

repo_id = "tu-usuario/pibot-jointbert"

# Tokenizer y config
tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
config = AutoConfig.from_pretrained(repo_id, trust_remote_code=True)

# Training args
args_path = hf_hub_download(repo_id, "training_args.bin")
train_args = torch.load(args_path, weights_only=False)

# Labels
def read_labels(repo_id, filename):
    path = hf_hub_download(repo_id, f"labels/{filename}")
    return [l.strip() for l in open(path, encoding="utf-8") if l.strip()]

label_names = ["calc_mode", "activity", "region", "investment", "req_form", "slot"]
labels = {name: read_labels(repo_id, f"{name}_label.txt") for name in label_names}

# Modelo
model = AutoModel.from_pretrained(
    repo_id,
    config=config,
    trust_remote_code=True,
    args=train_args,
    calc_mode_label_lst=labels["calc_mode"],
    activity_label_lst=labels["activity"],
    region_label_lst=labels["region"],
    investment_label_lst=labels["investment"],
    req_form_label_lst=labels["req_form"],
    slot_label_lst=labels["slot"],
)
model.eval()
```

### Uso en un endpoint (ej. Azure ML, FastAPI)

```python
from huggingface_hub import snapshot_download
from pathlib import Path

# Descargar todo el checkpoint
model_dir = snapshot_download("tu-usuario/pibot-jointbert")

# Leer labels desde carpeta local
labels_dir = Path(model_dir) / "labels"
labels = {}
for f in labels_dir.glob("*_label.txt"):
    labels[f.stem] = [line.strip() for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]

# Luego cargar tokenizer + modelo como arriba
```

---

## Resumen del flujo

```text
1. Entrenar modelo      →  model_out/{nombre}/
2. Copiar a package     →  model_package/  (pesos + tokenizer + código + labels)
3. Validar y subir      →  python util/upload_model_package.py --repo_name ...
4. Consumir desde HF    →  snapshot_download("usuario/repo") o AutoModel.from_pretrained(...)
```
