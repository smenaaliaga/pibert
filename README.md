# Clasificador Joint BERT para PIBot

Implementación de un modelo **joint intent + slot filling** para consultas macroeconómicas (PIBERT), basado en el enfoque de JointBERT.

- Paper base: https://arxiv.org/abs/1810.04805v2
- Implementación de referencia: https://github.com/monologg/JointBERT

## ¿Qué resuelve este proyecto?

Dada una consulta en lenguaje natural, el modelo predice simultáneamente:

1. **5 cabezas de intención**:
   - `calc_mode`
   - `activity`
   - `region`
   - `investment`
   - `req_form`
2. **Etiquetas de entidades (slots)** en formato BIO (`B-*`, `I-*`, `O`), por token.

Entre las entidades NER se incluye `PRICE` (`B-PRICE`, `I-PRICE`), que identifica menciones de precios en el texto.

## Arquitectura (resumen)

- Encoder Transformer (BERT/BETO/DeBERTa/RoBERTa).
- Cabezas de clasificación independientes para cada intención.
- Cabeza de secuencia para slot filling (opcional con `--use_crf`).

### Modelos soportados (`--model_type`)

| Shortcut | Modelo |
|---|---|
| `bert` | `bert-base-uncased` |
| `beto` | `dccuchile/bert-base-spanish-wwm-cased` |
| `deberta` | `microsoft/mdeberta-v3-base` |
| `roberta` | `bertin-project/bertin-roberta-base-spanish` |

## Requisitos

- Python 3.10+
- `pip`
- GPU NVIDIA opcional (recomendado para entrenamiento)

## Instalación rápida

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

> Nota: `requirements.txt` está preparado para PyTorch con CUDA 12.1.

## Estructura de datos esperada

Para cada dataset en `data/{dataset}`:

```text
data/{dataset}/
├─ raw/                 # Fuente original (ej. Excel)
├─ processed/           # Archivos consolidados para split
│  ├─ seq.in
│  ├─ label
│  └─ seq.out
├─ train/
├─ dev/
├─ test/
├─ calc_mode_label.txt
├─ activity_label.txt
├─ region_label.txt
├─ investment_label.txt
├─ req_form_label.txt
└─ slot_label.txt
```

### Formato de archivos clave

- `seq.in`: una consulta por línea.
- `label`: 5 etiquetas separadas por coma por línea, en este orden:
  `calc_mode,activity,region,investment,req_form`
- `seq.out`: etiquetas BIO por token separadas por espacio.

## Flujo de trabajo recomendado

### 1) Procesar Excel a formato JointBERT

```bash
python util/process_raw_data.py --dataset pibimacecv3 --input_file data/pibimacecv3/raw/dataset.xlsx
```

Columnas esperadas en el Excel (formato nuevo recomendado):
`utterance`, `calc_mode` (o `calc_mode_cls`), `activity` (o `activity_cls`),
`region` (o `region_cls`), `investment` (o `investment_cls`),
`req_form` (o `req_form_cls`), `ner_bio`.

Compatibilidad legacy:
`Utterance`, `CalcMode`, `ActivityCls`, `RegionCls`, `InvestmentCls`, `ReqForm`, `NER_BIO`.

Nota: `ner_bio` / `NER_BIO` es opcional; si no existe, se generan etiquetas `O` para `seq.out`.

Esto genera:
- `data/{dataset}/processed/{seq.in,label,seq.out}`
- `data/{dataset}/*_label.txt`

### 2) Dividir train/dev/test

```bash
python util/split_data.py --dataset pibimacecv3 --test_size 0.1 --dev_size 0.1 --seed 42
```

### 3) Entrenar

Ejemplo base (BETO):

```bash
python main.py --task pibimacecv3 --model_type beto --model_out pibert --do_train --do_eval
```

Ejemplo recomendado (ajustado):

```bash
python main.py --task pibimacecv3 --model_type beto --model_out pibert --do_train --do_eval --max_seq_len 64 --train_batch_size 8 --gradient_accumulation_steps 2 --eval_batch_size 32 --learning_rate 1.5e-5 --num_train_epochs 10 --warmup_steps 300 --weight_decay 0.02 --dropout_rate 0.25 --logging_steps 100 --save_steps 300 --early_stopping --early_stopping_patience 2 --seed 42
```

Con CRF:

```bash
python main.py --task pibimacecv3 --model_type beto --model_out pibert_crf --do_train --do_eval --use_crf --use_class_weights --max_seq_len 64 --train_batch_size 8 --gradient_accumulation_steps 2 --eval_batch_size 32 --learning_rate 1.5e-5 --num_train_epochs 10 --warmup_steps 300 --weight_decay 0.02 --dropout_rate 0.25 --logging_steps 100 --save_steps 300 --early_stopping --early_stopping_patience 2 --seed 42
```

```bash
python main.py --task pibimacecv3 --model_type deberta --model_out pibert_deberta_crf --do_train --do_eval --use_crf --use_class_weights --max_seq_len 64 --train_batch_size 8 --gradient_accumulation_steps 2 --eval_batch_size 32 --learning_rate 1.5e-5 --num_train_epochs 10 --warmup_steps 300 --weight_decay 0.02 --dropout_rate 0.25 --logging_steps 100 --save_steps 300 --early_stopping --early_stopping_patience 2 --seed 42
```

> Estas configuraciones priorizan generalización (más regularización y menos sobreentrenamiento). Ajusta `warmup_steps` a ~8-10% de los pasos totales de optimización y elige por `slot_f1` + `semantic_frame_acc` en test.

> `main.py` guarda y carga modelos bajo `model_out/{model_out}` automáticamente.

## Predicción

Este repositorio incluye `predict_cli.py` para inferencia interactiva, por texto único y por archivo.

### Modo interactivo

```bash
python predict_cli.py --model_dir model_out/pibert_deberta_crf
```

### Texto único

```bash
python predict_cli.py --model_dir model_out/pibert_deberta_crf --text "cual fue el ultimo imacec"
```

### Batch por archivo

```bash
python predict_cli.py --model_dir model_out/pibert_base --input_file consultas.txt --output_file resultados.txt
```

Salida incluye:
- predicción y confianza por cada cabeza
- entidades agrupadas detectadas desde BIO

## Argumentos principales de entrenamiento

| Argumento | Descripción | Default |
|---|---|---|
| `--task` | Nombre del dataset en `data/` | requerido |
| `--model_out` | Nombre de salida (dentro de `model_out/`) | requerido |
| `--model_type` | Shortcut del modelo base | `bert` |
| `--max_seq_len` | Longitud máxima de secuencia | `50` |
| `--train_batch_size` | Batch de entrenamiento | `32` |
| `--eval_batch_size` | Batch de evaluación | `64` |
| `--learning_rate` | Learning rate Adam | `5e-5` |
| `--num_train_epochs` | Épocas | `10` |
| `--dropout_rate` | Dropout en capas FC | `0.1` |
| `--early_stopping` | Activa early stopping | `False` |
| `--use_crf` | Activa capa CRF para slots | `False` |

## Estructura del proyecto

```text
bc_pibot_jointbert/
├─ data/
├─ model_in/                 # Definición de JointBERT
├─ model_out/                # Checkpoints y modelos entrenados
├─ util/
│  ├─ split_data.py
│  ├─ upload_to_hf.py
│  └─ upload_to_hf_improved.py
├─ data_loader.py
├─ main.py                   # Entrenamiento/evaluación
├─ predict_cli.py            # Inference CLI
├─ process_raw_data.py       # Procesamiento de Excel
├─ trainer.py
├─ utils.py
├─ requirements.txt
└─ README.md
```

## Publicar en Hugging Face

### 1. Login en Hugging Face

```bash
hf auth login
```

### 2. Preparar + subir modelo a HF

```bash
python util/prepare_model_package.py --model_dir model_out/{NOMBRE_MODELO} --labels_dir data/{NOMBRE_DATASET} --clean --upload --repo_name BCCh/pibert
```

- `--model_dir`: carpeta del checkpoint entrenado a publicar.
- `--labels_dir`: carpeta con los archivos `*_label.txt` que usa el modelo.
- `--clean`: limpia `model_package/` antes de copiar (evita mezclar archivos viejos).
- `--upload`: sube automáticamente después de preparar el paquete.
- `--repo_name`: repositorio destino en HF (`usuario-o-org/nombre-repo`).

### 3. Utilizar modelo

Una vez subido, otros proyectos pueden usarlo así:

```python
from huggingface_hub import snapshot_download
model_dir = snapshot_download("tu-usuario/pibot-jointbert")
```

O directamente con `transformers`:

```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("tu-usuario/pibot-jointbert", trust_remote_code=True)
```

> Ver `model_package/README.md` para instrucciones completas de carga e inferencia.
