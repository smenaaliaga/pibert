# Clasificador Joint BERT para PIBot 

Paper DOI: [BERT for Joint Intent Classification and Slot Filling](https://arxiv.org/abs/1810.04805v2) \
Resumen paper: [docs/BERT_for_Joint_Intent_Classification_and_Slot_Filling.md](docs/BERT_for_Joint_Intent_Classification_and_Slot_Filling.md) \
Implementación adaptada de: https://github.com/monologg/JointBERT

## Objetivo

Entrenar un modelo de clasificación multi-cabeza basado en arquitecturas Transformer (BERT, RoBERTa, DeBERTa) para detectar las series de la Base de Datos Estadisticos (BDE) del Banco Central de Chile (BCCh) utizado para el chatbot - PIBot. El modelo procesa consultas economicas y clasifica simultáneamente en 5 diferentes intenciones y hasta 4 entidades necesarias para identificar una serie economica.

## Arquitectura Multi-Cabeza

El modelo se compone **cinco cabezas de clasificacion de intencion y una de reconocimiento de entidades** que procesan en paralelo diferentes aspectos de la consulta.

### Intent Classification

| Cabezas | Valores posibles | Descripción |
|--------|------------------|-------------|
| `calc_mode` | `original`, `prev_period`, `yoy`, `contribution` | Tipo de cálculo solicitado |
| `frequency` | `m`, `q`, `a` | Frecuencia temporal de la serie |
| `activity` | `general`, `specific`, `none` | Presencia de actividad económica en la consulta |
| `region` | `general`, `specific`, `none` | Presencia de region en la consulta |
| `req_form` | `latest`, `point`, `range` | Formato del periodo consultaod |

### Slot Filling - Reconocimiento de Entidades

El módulo de **Slot Filling** utiliza el esquema de etiquetado **BIO (Begin-Inside-Outside)** para identificar y clasificar entidades en la consulta a nivel de token:

| Etiqueta | Descripción | Ejemplos |
|----------|-------------|----------|
| `B-INDICATOR`, `I-INDICATOR`  | Indicadores económicos mencionados | PIB o IMACEC |
| `B-SEASONALITY`, `I-SEASONALITY` | Información sobre ajuste estacional | Estacional (nsa) o desestacionalizado (nsa) |
| `B-ACTIVITY`, `I-ACTIVITY` | Sectores o actividades económicas | Minería, Industria, Servicios, etc |
| `B-REGION`, `I-REGION` | Ubicaciones geográficas | Metropolitana, Valparaíso, Bío Bío, etc |
| `O` (Other) | Tokens sin entidad específica | conjunciones, preposiciones, etc. |

### Ejemplo practico

```
Entrada: "Cuánto vario la actividad no minera del PIB el último trimestre?"

→ BERT encoder

├─ calc_mode: yoy       # Var. mismo periodo respecto año anterior
├─ frequency: q         # Frecuencia trimestral
├─ activity: specific   # Presencia de actividad economica especifica
├─ region: none         # Sin presencia de región en la consulta
├─ req_form: latest     # Consulta del último periodo disponible
└─ slots: [O, O, O, O, B-ACTIVITY, I-ACTIVITY, O, B-INDICATOR, O, O, O]
```


## Modelos Soportados

Modelos soportados para el **re-entrenamiento** (HuggingFace). Estos *shortcuts* permiten configurar rápidamente el encoder base:

| Shortcut | Modelo | Descripción |
|----------|--------|-------------|
| `bert` | `bert-base-uncased` | BERT base inglés |
| `beto` | `dccuchile/bert-base-spanish-wwm-cased` | BETO - BERT español |
| `deberta` | `microsoft/mdeberta-v3-base` | DeBERTa v3 multilingüe |
| `roberta` | `PlanTL-GOB-ES/roberta-base-bne` | RoBERTa español |

## Flujo de Trabajo Completo

### 0. Estructura de Datasets

Todos los datasets se almacenan en la carpeta `/data`. Cada dataset debe seguir la siguiente estructura de subdirectorios:

```
data/
└── {nombre_dataset}/
    ├── raw/          # Datos procesados (seq.in, label, seq.out)
    ├── train/        # Conjunto de entrenamiento
    ├── dev/          # Conjunto de desarrollo/validación
    └── test/         # Conjunto de prueba
```

**Nota:** Las carpetas `train/`, `dev/` y `test/` se generan automáticamente al ejecutar el script de división de datos (ver sección 2).

### 1. Preparación de Datos Crudos

Los datos deben prepararse en la carpeta `data/{dataset}/processed/` con los siguientes archivos en formato de texto plano:

- **`seq.in`**: Textos de entrada (una consulta por línea)
- **`label`**: Etiquetas de intención separadas por comas (5 valores por línea: `calc_mode,frequency,activity,region,req_form`)
- **`seq.out`**: Etiquetas de entidades en formato BIO (una secuencia por línea, tokens separados por espacios)

**Ejemplo:**

`seq.in`:
```
cual fue el pib del ultimo trimestre
```

`label`:
```
yoy,q,general,none,latest
```

`seq.out`:
```
O O O B-INDICATOR O O O O
```

### 2. División de Datos en Train/Dev/Test

Una vez procesados los datos, divídelos en conjuntos de entrenamiento, desarrollo y prueba:

```bash
python util/split_data.py --dataset pibimacecv4 --test_size 0.2 --dev_size 0.1 --seed 42
```

**Parámetros:**
- `--dataset`: Nombre del dataset
- `--test_size`: Proporción para test (default: 0.2 = 20%)
- `--dev_size`: Proporción para dev (default: 0.1 = 10%)
- `--seed`: Semilla aleatoria para reproducibilidad (default: 42)

**Estructura generada:**
```
data/{dataset}/
├── train/
│   ├── label
│   ├── seq.in
│   └── seq.out
├── dev/
│   ├── label
│   ├── seq.in
│   └── seq.out
├── test/
│   ├── label
│   ├── seq.in
│   └── seq.out
├── raw/          # Datos crudos originales
├── *_label.txt
```

### 3. Entrenamiento del Modelo

#### Configuración Básica

```bash
# Usando shortcut
python main.py --task pibimacecv3 --model_type beto --model_out pibot_model_v3 --do_train --do_eval

# Usando modelo directamente (recomendado para mayor flexibilidad)
python main.py --task pibimacecv3 --model_type beto --model_name_or_path dccuchile/bert-base-spanish-wwm-cased --model_out pibot_model_v3 --do_train --do_eval

# Usando DeBERTa v3
python main.py --task pibimacecv3 --model_type deberta --model_out pibot_model_deberta --do_train --do_eval

# Usando RoBERTa español
python main.py --task pibimacecv3 --model_type roberta --model_out pibot_model_roberta --do_train --do_eval
```

#### Recomendación actual

```bash
python main.py --task pibimacecv3 --model_type beto --model_out pibot_model_v3 --do_train --do_eval --max_seq_len 64 --train_batch_size 8 --eval_batch_size 32 --learning_rate 1.5e-5 --num_train_epochs 15 --warmup_steps 150 --weight_decay 0.01 --dropout_rate 0.2 --logging_steps 50 --save_steps 100 --early_stopping --early_stopping_patience 3 --seed 42
```

**Parámetros principales:**

| Parámetro | Descripción | Recomendado | Nota |
|-----------|-------------|-------------|------|
| `--task` | Nombre del dataset | `pibimacecv3` | |
| `--model_type` | Shortcut del modelo | `beto`, `bert`, `deberta`, `roberta` | Usa shortcuts de MODEL_PATH_MAP |
| `--model_name_or_path` | Ruta o ID de HuggingFace del modelo | - | Opcional: sobrescribe model_type |
| `--model_out` | Nombre del modelo guardado | `pibot_model_v3` | Se guarda en model_out/{nombre} |
| `--max_seq_len` | Longitud máxima de secuencia | `64` | Consultas cortas en español |
| `--train_batch_size` | Tamaño de batch para entrenamiento | `8` | Más actualizaciones por época (375 pasos) |
| `--eval_batch_size` | Tamaño de batch para evaluación | `32` | |
| `--learning_rate` | Tasa de aprendizaje | `1.5e-5` | Conservador para evitar overfitting |
| `--num_train_epochs` | Número de épocas | `15` | Más épocas con dataset pequeño |
| `--weight_decay` | Regularización L2 | `0.01` | Regularización moderada |
| `--warmup_steps` | Pasos de warmup | `100` | ~10% del total de pasos |
| `--dropout_rate` | Tasa de dropout | `0.2` | Mayor regularización para dataset pequeño |
| `--logging_steps` | Frecuencia de logs | `50` | Monitoreo frecuente |
| `--save_steps` | Frecuencia de checkpoints | `100` | Guardar modelos intermedios |
| `--early_stopping` | Detener cuando no mejore la pérdida de dev | `True` | Activa parada temprana por `val_loss` |
| `--early_stopping_patience` | Épocas sin mejora antes de parar | `3` | Guarda el mejor modelo al mejorar |

## Prediction

### Predicción en lote (predict.py)

Para procesar múltiples consultas desde un archivo:

```bash
python predict.py --input_file {INPUT_FILE_PATH} --output_file {OUTPUT_FILE_PATH} --model_dir {SAVED_CKPT_PATH}

# Ejemplo:
python predict.py --input_file data/atis/test/seq.in --output_file output/atis.txt --model_dir model_out/atis_model
```

### Predicción desde línea de comandos (predict_cli.py)

#### Modo interactivo
Permite hacer consultas en tiempo real desde la terminal:

```bash
python predict_cli.py --model_dir {SAVED_CKPT_PATH}

# Ejemplo:
python predict_cli.py --model_dir model_out/pibot_model_v3
```

**Salida del modelo:**

El modelo retorna simultáneamente:

1. **Clasificaciones de intención** (5 cabezas):
   - `calc_mode`: Tipo de cálculo solicitado
   - `frequency`: Frecuencia temporal
   - `activity`: Actividad económica
   - `region`: Región geográfica
   - `req_form`: Período solicitado

2. **Etiquetas de entidades** (Slot Filling):
   - Token-level tagging: `INDICATOR`, `SEASONALITY`, `ACTIVITY`, `REGION`, `O`
   - Confianza (softmax) para cada token

**Ejemplo de salida:**

```
============================================================
calc_mode     : yoy                 (confidence: 94.23%)
frequency     : q                   (confidence: 87.15%)
activity      : total               (confidence: 92.45%)
region        : nacional            (confidence: 88.92%)
req_form      : latest              (confidence: 91.67%)
============================================================

Entidades detectadas:
  - INDICATOR: [IMACEC, PIB]
  - SEASONALITY: [desestacionalizado]
  - ACTIVITY: [minería, construcción]
  - REGION: [metropolitana]
```

## Instalación y Dependencias

### Requisitos básicos

- Python >= 3.12 y pip

### Instalación

1. **Crear entorno virtual:**

```bash
python -m venv .venv
```

2. **Activar el entorno virtual:**

```bash
# Windows
.venv\Scripts\Activate.ps1
# Linux
source .venv/bin/activate
```

3. **Instalar dependencias:**

```bash
pip install -r requirements.txt
```

### Dependencias Principales

#### Bibliotecas principales

| Paquete | Versión | Propósito |
|---------|---------|-----------|
| `torch` | 2.5.1+cu121 | Framework de deep learning (GPU) |
| `transformers` | 4.46.0 | Modelos BERT pre-entrenados |
| `pytorch-crf` | 0.7.2 | Soporte CRF (actualmente no usado) |
| `seqeval` | 1.2.2 | Métricas para evaluación NER |
| `tqdm` | 4.67.1 | Barras de progreso |
| `numpy` | 2.3.3 | Operaciones numéricas |
| `pandas` | >=2.0.0 | Procesamiento de datos (Excel) |
| `openpyxl` | >=3.0.0 | Lectura de archivos Excel |
| `dateparser` | >=1.1.0 | Parsing de fechas |
| `rapidfuzz` | >=3.0.0 | Fuzzy matching |

Para ver todas las dependencias organizadas por categoría, consulta [requirements.txt](requirements.txt).

#### GPU (CUDA 12.1+)
Soporte GPU NVIDIA RTX 4080 Super, se requiere PyTorch con CUDA:

```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
```

**Verificar compatibilidad CUDA:**
```python
import torch
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

## Estructura del Proyecto

```
bc_pibot_jointbert/
├── data/                      # Datasets
│   └── {dataset_name}/
│       ├── raw/              # Datos crudos (Excel)
│       ├── processed/        # Datos procesados
│       ├── train/            # Conjunto de entrenamiento
│       ├── dev/              # Conjunto de desarrollo
│       ├── test/             # Conjunto de prueba
│       └── *_label.txt       # Vocabularios por cabeza
├── model_in/                  # Definición del modelo
│   ├── __init__.py
│   ├── modeling_jointbert.py # Modelo principal
│   └── module.py             # Cabezas clasificadoras
├── model_out/                 # Modelos entrenados
├── util/                      # Scripts de utilidad
│   ├── process_raw_data.py   # Procesar Excel
│   └── split_data.py         # Dividir datos
├── docs/                      # Documentación
├── main.py                    # Script principal
├── trainer.py                 # Lógica de entrenamiento
├── data_loader.py             # Carga de datos
├── utils.py                   # Utilidades
├── predict.py                 # Predicción batch
├── predict_cli.py             # Predicción CLI
├── chat.py                    # Chat interactivo
├── requirements.txt           # Dependencias
└── README.md                  # Este archivo
```

## Publicar en Hugging Face

Usa el script [util/upload_to_hf.py](util/upload_to_hf.py) para subir modelos entrenados a Hugging Face Hub.

### Requisitos previos

El script espera encontrar en `model_dir`:
- **Pesos**: `model.safetensors` o `pytorch_model.bin`
- **Config**: `config.json`
- **Tokenizer**: `tokenizer.json`, `tokenizer_config.json`, `vocab.txt`, `special_tokens_map.json` (se generan automáticamente tras entrenar si usaste `main.py`)
- **Código custom**: `modeling_jointbert.py`, `module.py`, `__init__.py`
- **Labels**: Los 7 archivos `*_label.txt`

El script puede **completar archivos faltantes** automáticamente:
- Si faltan labels, los copia desde `--label_dir`
- Si faltan archivos de código custom, los copia desde `--code_dir`

### Uso básico

```bash
python util/upload_to_hf.py --repo_name <usuario>/<nombre-repo> --model_dir <directorio-modelo> --label_dir <directorio-labels>
```

### Argumentos

| Argumento | Tipo | Requerido | Descripción |
|-----------|------|-----------|-------------|
| `--repo_name` | str | ✓ | Nombre del repo en HF (formato: `usuario/nombre-modelo`) |
| `--model_dir` | str | ✓ | Directorio del modelo a subir |
| `--label_dir` | str | ✓ | Directorio con los 7 `*_label.txt` (se copian si faltan en `model_dir`) |
| `--code_dir` | str | ✗ | Directorio con código custom (`modeling_jointbert.py`, `module.py`, `__init__.py`) para copiar si faltan en `model_dir` |
| `--token` | str | ✗ | Token de Hugging Face (si no se provee, usa el guardado o solicita login) |
| `--private` | flag | ✗ | Crear repositorio privado |
| `--create_model_card` | flag | ✗ | Generar README.md automático en `model_dir` antes de subir |
| `--commit_message` | str | ✗ | Mensaje de commit (default: "Upload PIBot Joint BERT model") |
| `--no_login` | flag | ✗ | No intentar login interactivo (usar solo token existente) |

### Ejemplos

**Subida básica** (asumiendo que `model_dir` ya tiene todos los archivos):
```bash
python util/upload_to_hf.py --repo_name smenaaliaga/pibot-jointbert --model_dir model_out/pibot_model_v3 --label_dir data/pibimacecv3 --commit_message "Upload PIBot Joint BERT 7-heads model"
```
