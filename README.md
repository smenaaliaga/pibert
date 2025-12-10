# Clasificador de intención y palabras claves para PIBot usando Joint BERT

Paper DOI: [BERT for Joint Intent Classification and Slot Filling](https://arxiv.org/abs/1810.04805v2) \
Resumen paper: [docs/BERT_for_Joint_Intent_Classification_and_Slot_Filling.md](docs/BERT_for_Joint_Intent_Classification_and_Slot_Filling.md) \
Implementación adaptada de:  https://github.com/monologg/JointBERT

## Objetivo

El objetivo es entrenar un modelo de clasificación conjunta de intención y palabras claves para ser utilizado por el chatbot del Banco Central de Chile (PIBot). 

La clasificación está separada en dos partes, entender la intención de la consulta, ¿se pregunta por valores o metodología? e, identificar las palabras claves de la consulta: la métrica (PIB o IMACEC), el periodo, tipo de agregación, entre otros.

## Dataset

El principal objetivo de este proyecto es entrenar el modelo Joint BERT con un dataset especializado para detectar la intención y las palabras claves para consultas sobre PIB e IMACEC.

### label

TODO

### Definición de BIO

La definición de la BIO propuesta es la siguiente:

#### 1. Indicador Económico (el "qué")

Para preguntas como “cuál fue el IMACEC”, “cómo evolucionó el PIB trimestral”, etc.
- B-indicator
- I-indicator

Ejemplos de spans:
- IMACEC
- PIB
- crecimiento económico
- actividad económica

...

### Anotación con Label Studio

Para anotar datos de entrenamiento, el proyecto utiliza Label Studio. A continuación se detallan los pasos para ejecutarlo:

#### Iniciar Label Studio

```bash
label-studio start
```

Esto abrirá Label Studio en tu navegador (por defecto en `http://localhost:8080`).

#### Configuración del proyecto

1. Crea un nuevo proyecto en Label Studio
2. Importa tus datos (archivos de texto con consultas)
3. Configura la interfaz de anotación para:
   - **Clasificación de intención**: prev_year, calc_mode, etc.
   - **Etiquetado de entidades (NER)**: indicator, period, measure_type, sector, etc.

#### Exportación de datos

Una vez anotados los datos:
1. Exporta en formato CSV desde Label Studio
2. Utiliza el script de procesamiento:

```bash
python util/process_raw_labels.py data/pibimacec/raw/annotations.csv data/pibimacec/processed
```

Este script convierte las anotaciones de Label Studio al formato BIO requerido por Joint BERT.

### División de datos

Para crear los conjuntos de entrenamiento, desarrollo y prueba:

```bash
python util/split_data.py data/pibimacec/processed data/pibimacec --test_size 0.2 --dev_size 0.1
```

## Training & Evaluation

```
python main.py --task {task_name} \
                  --model_type {model_type} \
                  --model_out {model_out_name} \
                  --do_train --do_eval \
                  --use_crf
```
Ejemplo básico:
```bash
python main.py --task pibimacec --model_type beto --model_out pibot_model_beto --do_train --do_eval
```

Configuración optimizada para dataset pibimacec (816 ejemplos):
```bash
python main.py --task pibimacec --model_type beto --model_out pibot_model_beto_largo --do_train --do_eval --num_train_epochs 20 --learning_rate 3e-5 --train_batch_size 16 --eval_batch_size 32 --max_seq_len 64 --save_steps 20 --logging_steps 10 --ignore_index -100
```

Parámetros justificados:
- `--num_train_epochs 20`: Mayor número de épocas para dataset pequeño
- `--learning_rate 3e-5`: Tasa reducida para evitar overfitting
- `--train_batch_size 16`: Batch pequeño para más actualizaciones con pocos datos
- `--max_seq_len 32`: Secuencias cortas (promedio 10 palabras, máx 20)
- `--save_steps 20`: Checkpoints cada ~0.5 épocas


## Prediction

### Predicción en lote (predict.py)

Para procesar múltiples consultas desde un archivo:

```
python predict.py --input_file {INPUT_FILE_PATH} --output_file {OUTPUT_FILE_PATH} --model_dir {SAVED_CKPT_PATH}
```

Ejemplo:

```
python predict.py --input_file data/atis/test/seq.in --output_file output/atis.txt --model_dir model_out/atis_model
```

### Predicción desde línea de comandos (predict_cli.py)

#### Modo interactivo
Permite hacer consultas en tiempo real desde la terminal:

```bash
python predict_cli.py --model_dir model_out/pibot_model_beto
```

#### Consulta única
Para predecir un texto específico:

```bash
python predict_cli.py --model_dir model_out/pibot_model_beto --text "cual fue el ultimo imacec"
```

#### Modo batch
Para procesar múltiples consultas desde un archivo:

```bash
python predict_cli.py --model_dir model_out/pibot_model_beto --input_file consultas.txt --output_file resultados.txt
```

### Chat interactivo (chat.py)

Para una experiencia de chat más amigable con colores y formato mejorado:

```bash
python chat.py --model_dir model_out/pibot_model_beto
```

El chat muestra:
- Intención detectada con nivel de confianza
- Slots (palabras clave) identificados con sus etiquetas
- Formato visual con colores para mejor legibilidad

Para salir del chat, escribe `salir`, `exit` o presiona `Ctrl+C`.

## Resultados

## Dependencias

- python>=3.12

## TODO

- Adaptar el train para modelos BERT en español como BETO
- Crear datasets de preguntas con intención y slots
- Entrenar el modelo y realizar las pruebas