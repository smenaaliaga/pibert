# Clasificador Joint BERT para PIBot 

Paper DOI: [BERT for Joint Intent Classification and Slot Filling](https://arxiv.org/abs/1810.04805v2) \
Resumen paper: [docs/BERT_for_Joint_Intent_Classification_and_Slot_Filling.md](docs/BERT_for_Joint_Intent_Classification_and_Slot_Filling.md) \
Implementación adaptada de: https://github.com/monologg/JointBERT

## Objetivo

Entrenar un modelo de clasificación multi-cabeza basado en BERT para el chatbot del Banco Central de Chile (PIBot). El modelo procesa consultas en lenguaje natural y clasifica simultáneamente 7 aspectos diferentes de la consulta económica.

## Arquitectura Multi-Cabeza

El modelo utiliza **7 cabezas clasificadoras independientes** que procesan en paralelo diferentes aspectos de la consulta:

| Cabeza | Valores posibles | Descripción |
|--------|------------------|-------------|
| `indicator` | `imacec`, `pib` | Indicador económico |
| `metric_type` | `index`, `contribution` | Tipo de metrica |
| `seasonality` | `sa`, `nsa` | Ajuste estacional |
| `activity` | `total`, `imc_*`, `pib_*` | Sector/actividad |
| `frequency` | `m`, `q`, `a` | Frecuencia temporal |
| `req_form` | `latest`, `point`, `range` | Forma de la solicitud |
| `calc_mode` | `none`, `yoy`, `prev_period` | Tipo de variación |

Cada cabeza opera de forma independiente sobre la representación contextual del texto generada por BERT, permitiendo clasificación multi-etiqueta eficiente.

## Flujo de Trabajo Completo

### 1. Preparación de Datos Crudos

#### 1.1 Formato del Excel de Entrada

El archivo Excel debe contener las siguientes columnas en orden:

Coloca el Excel de datos crudos dentro de la carpeta `data/{dataset}/raw/` (ejemplo: `data/pibimacecv3/raw/data.xlsx`).

| Columna | Nombre | Descripción | Ejemplo |
|---------|--------|-------------|---------|
| 1 | text | Texto de la consulta | "cual fue el pib del ultimo trimestre" |
| 2 | indicator | Indicador económico | "pib" |
| 3 | metric_type | Tipo de métrica | "index" |
| 4 | seasonal | Ajuste estacional | "sa" |
| 5 | activity | Tipo de actividad | "total" |
| 6 | frequency | Frecuencia | "q" |
| 7 | req_form | Forma de solicitud | "latest" |
| 8 | calc_mode | Modo de cálculo | "yoy" |

#### 1.2 Procesamiento de Datos Crudos

Utiliza `process_raw_data.py` para convertir el Excel a los archivos de entrenamiento:

```bash
python util/process_raw_data.py --dataset pibimacecv3 --input_file data/pibimacecv3/raw/data.xlsx --sheet_name final
```

**Parámetros:**
- `--dataset`: Nombre del dataset (crea carpeta en `data/`)
- `--input_file`: Ruta al archivo Excel (.xlsx o .xls)
- `--sheet_name`: (Opcional) Nombre de la hoja del Excel
- `--no_skip_header`: (Opcional) No saltar primera fila si no es encabezado

**Archivos generados en `data/{dataset}/processed/`:**
- `seq.in`: Textos de entrada (uno por línea)
- `label`: Etiquetas por cabeza separadas por comas
- `seq.out`: Etiquetas de slot (solo "O" por token, no usado en esta versión)

**Archivos generados en `data/{dataset}/`:**
- `indicator_label.txt`: Vocabulario de indicadores
- `metric_type_label.txt`: Vocabulario de tipos de métrica
- `calc_mode_label.txt`: Vocabulario de modos de cálculo
- `seasonal_label.txt`: Vocabulario de ajustes estacionales
- `req_form_label.txt`: Vocabulario de formas de solicitud
- `frequency_label.txt`: Vocabulario de frecuencias
- `activity_label.txt`: Vocabulario de actividades
- `slot_label.txt`: Vocabulario de slots (solo contiene "O")

**Ejemplo de salida:**

`data/pibimacecv3/processed/seq.in`:
```
cual fue el pib del ultimo trimestre
cuanto acelero el imacec minero en agosto
```

`data/pibimacecv3/processed/label`:
```
pib,index,sa,total,q,latest,yoy
imacec,variation,sa,minero,m,point,prev_period
```

### 2. División de Datos en Train/Dev/Test

Una vez procesados los datos, divídelos en conjuntos de entrenamiento, desarrollo y prueba:

```bash
python util/split_data.py --dataset pibimacecv3 --test_size 0.2 --dev_size 0.1 --seed 42
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
├── processed/    # Datos originales procesados
├── raw/          # Datos crudos originales
├── *_label.txt
```

**Distribución típica:**
- Train: 70% (560 ejemplos de 800 totales)
- Dev: 10% (80 ejemplos)
- Test: 20% (160 ejemplos)

### 3. Entrenamiento del Modelo

#### Configuración Básica

```bash
python main.py --task pibimacecv3 --model_type beto --model_out pibot_model_v3 --do_train --do_eval
```

#### Recomendación actual

```bash
python main.py --task pibimacecv3 --model_type beto --model_out pibot_model_v3 --do_train --do_eval --max_seq_len 64 --train_batch_size 8 --eval_batch_size 32 --learning_rate 1.5e-5 --num_train_epochs 15 --warmup_steps 150 --weight_decay 0.01 --dropout_rate 0.2 --logging_steps 50 --save_steps 100 --early_stopping --early_stopping_patience 3 --seed 42
```

**Parámetros principales:**

| Parámetro | Descripción | Recomendado | Nota |
|-----------|-------------|-------------|------|
| `--task` | Nombre del dataset | `pibimacecv3` | |
| `--model_type` | Modelo BERT base | `beto` (español) o `bert` (inglés) | BETO recomendado para español |
| `--model_out` | Nombre del modelo guardado | `pibot_model_v3` | |
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

#### Comando recomendado para dataset pequeño:

#### Alternativas según rendimiento:

**Si hay overfitting (val loss ↑ después de epoch ~10):**
```bash
--train_batch_size 16 --learning_rate 1e-5 --num_train_epochs 20 --dropout_rate 0.25
```

**Si hay underfitting (val loss no mejora después de epoch 15):**
```bash
--train_batch_size 8 --learning_rate 2e-5 --num_train_epochs 30 --dropout_rate 0.15
```

**Para clases desbalanceadas (ej: activity con "no_minero" bajo representado):**
```bash
--learning_rate 1e-5 --num_train_epochs 30 --dropout_rate 0.25
```

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
