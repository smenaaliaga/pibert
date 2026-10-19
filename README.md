# PIBERT: de la consulta en lenguaje natural a la serie de la BDE

Sistema de comprensión de consultas sobre PIB e Imacec del Banco Central de Chile. Una consulta
se convierte en una **tupla parcial** de dimensiones (indicador, frecuencia, estacionalidad,
presentación, valoración, actividad, región, gasto, más período y cálculo) que se resuelve contra
el **catálogo** de series publicadas. El modelo neuronal es sólo el extractor; la resolución es
simbólica.

```
texto ──► extractor (JointBERT: cabezas + NER) ──► normalización ──► tupla parcial q
                                                                        │
                       catálogo (data/catalogo/series.csv) ◄────────────┘
                       filtrar → propagar restricciones → preferencias → serie | aclarar | no existe
```

- Paper base de la arquitectura de extracción: JointBERT, https://arxiv.org/abs/1902.10909
- Implementación de referencia: https://github.com/monologg/JointBERT

## Estado

| Componente | Estado |
|---|---|
| Catálogo de resolución (`util/build_catalog.py`, `data/catalogo/`) | listo, 4.490 series / 276 cuadros |
| Resolvedor (`catalog.py`) | listo: filtrado, propagación, preferencias, pregunta aclaratoria, cardinalidad por eje (Cuadro 2) |
| Modelo dirigido por esquema (`schema.py`, `model_in/`) | listo |
| Dataset con la definición del paper (3 cabezas binarias, 10 entidades) | `pibimacecv5`; 153 filas marcadas `Revisar` en `raw/dataset.xlsx` |
| Normalizadores (span → valor canónico) y `extract.py` | CALCULO y REFERENCIA en `normalizacion.py`; el resto pendiente |
| Evaluación de punta a punta (exactitud de tupla y de serie) | pendiente |
| Publicación en Hugging Face (`util/prepare_model_package.py`) | desactualizado respecto del nuevo formato de checkpoint |

## Instalación

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
# o: python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests
```

Con GPU NVIDIA instala `torch` primero según https://pytorch.org/get-started/locally/. En Apple
Silicon se usa MPS automáticamente.

## Catálogo

```bash
.venv/bin/python util/build_catalog.py     # regenera data/catalogo/{series,cuadros}.csv, vocabularios.json, reporte.md
```

```python
from catalog import Catalog
r = Catalog.load().resolve(region="valparaiso", actividad="comercio")
r.estado, r.cuadros, r.supuestos, r.colapsadas, r.aclarar
```

Detalles del formato de los códigos de serie, las tres clases de dimensión y las preferencias en
`data/catalogo/README.md`.

## Esquema del modelo

`schema.json` define las cabezas clasificadoras (nombre → etiquetas) y los tipos de entidad. Es la
única fuente de verdad para el modelo, el cargador de datos, el entrenador y la inferencia.

```bash
.venv/bin/python util/make_schema.py --task pibimacecv3 --from-data      # esquema heredado del dataset
.venv/bin/python util/make_schema.py --task pibimacecv5 --from-catalog   # esquema del paper desde vocabularios.json
```

Esquema del paper (sección 4): cabezas binarias `activ_cls`, `reg_cls`, `gast_cls`; entidades
INDICADOR, FRECUENCIA, ESTACIONALIDAD, PRESENTACION, VALORACION, ACTIVIDAD, REGION, GASTO, PERIODO,
CALCULO. Los spans de CALCULO se normalizan con `normalizacion.normalizar_calculo` (sin mención:
`nivel`); la referencia se resuelve fuera del modelo con `normalizacion.detectar_referencia`.

Las cabezas indican si la consulta invoca la clasificación completa de un eje; junto con los spans
fijan la cardinalidad de la respuesta en `Catalog.resolve_consulta`:

```python
r = Catalog.load().resolve_consulta({"activ_cls": 1}, indicador="imacec", estacionalidad="original")
r.modo         # "agregada" | "categorias" | "cuadro"
r.destacadas   # categorías mencionadas cuando el eje se pide completo
```

`pibimacecv5` se genera desde `pibimacecv4` con `util/migrate_v4_to_v5.py` (pre-anotación de CALCULO
por léxico; las filas cuyo cálculo esperado no coincide con sus spans quedan con `Revisar` y `Motivo`).

## Datos

```
data/<task>/
├─ raw/                 # Excel de origen
├─ processed/           # seq.in, label, seq.out consolidados
├─ train/ dev/ test/    # seq.in, label, seq.out
└─ schema.json
```

- `seq.in`: una consulta por línea.
- `label`: una etiqueta por cabeza separada por coma, en el orden de `schema.heads`.
- `seq.out`: etiquetas BIO por palabra.

Cualquier etiqueta fuera del esquema detiene la carga con un error que indica la línea.

```bash
.venv/bin/python util/process_raw_data.py --dataset pibimacecv3 --input_file data/pibimacecv3/raw/dataset.xlsx
.venv/bin/python util/split_data.py --dataset pibimacecv3 --test_size 0.15 --dev_size 0.1 --seed 42
```

## Entrenamiento

```bash
.venv/bin/python main.py --task pibimacecv3 --model_type beto --model_out pibert --do_train --do_eval \
  --max_seq_len 64 --train_batch_size 16 --learning_rate 2e-5 --num_train_epochs 10 --warmup_steps 300 \
  --weight_decay 0.01 --dropout_rate 0.2 --early_stopping --early_stopping_patience 2 --seed 42
```

| Argumento | Descripción | Default |
|---|---|---|
| `--task` | dataset en `data/` | requerido |
| `--model_out` | nombre del checkpoint dentro de `model_out/` | requerido |
| `--model_type` | `beto`, `deberta`, `roberta`, `bert`, `tiny` o un identificador de Hugging Face | `beto` |
| `--schema` | ruta a `schema.json` | `data/<task>/schema.json` |
| `--use_crf` | CRF sobre los slots | off |
| `--early_stopping` | guarda el mejor checkpoint por pérdida en dev; los guardados periódicos van a `model_out/<nombre>/last` | off |

El checkpoint es autocontenido: `config.json`, `model.safetensors`, `schema.json`, `jointbert.json`,
tokenizer y `training_args.json`. `--do_eval` escribe `eval_results.json` con accuracy y macro-F1 por
cabeza, P/R/F1 de slots y exactitud de frame semántico.

Prueba de humo (encoder de 4M parámetros, dos épocas en menos de un minuto):

```bash
.venv/bin/python main.py --task pibimacecv3 --model_type tiny --model_out smoke --do_train --do_eval \
  --num_train_epochs 2 --learning_rate 1e-3 --early_stopping --logging_steps 0
```

## Inferencia

```bash
.venv/bin/python predict_cli.py --model_dir model_out/pibert                         # interactivo
.venv/bin/python predict_cli.py --model_dir model_out/pibert --text "pib de valparaiso 2023"
.venv/bin/python predict_cli.py --model_dir model_out/pibert --input_file consultas.txt --output_file salida.jsonl
```

La salida trae etiqueta y confianza por cabeza, las etiquetas BIO por palabra y las entidades
agrupadas. La conversión de entidades a tupla parcial y la llamada a `Catalog.resolve` es el
siguiente paso pendiente.

## Estructura

```
pibert/
├─ catalog.py             # catálogo y resolvedor
├─ schema.py              # esquema de cabezas y entidades
├─ data_loader.py         # lectura y tensores
├─ trainer.py             # entrenamiento y evaluación
├─ main.py                # CLI de entrenamiento
├─ predict_cli.py         # CLI de inferencia
├─ utils.py               # métricas, tokenizer, dispositivo
├─ model_in/              # JointBERT dirigido por esquema
├─ util/                  # build_catalog, make_schema, process_raw_data, split_data, empaquetado HF
├─ data/                  # datasets y catálogo
├─ tests/
└─ paper/                 # borrador LaTeX
```
