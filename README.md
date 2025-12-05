# Clasificador de intención y palabras claves para PIBot usando Joint BERT

Paper DOI: [BERT for Joint Intent Classification and Slot Filling](https://arxiv.org/abs/1810.04805v2) \
Resumen paper: [docs/BERT_for_Joint_Intent_Classification_and_Slot_Filling.md](docs/BERT_for_Joint_Intent_Classification_and_Slot_Filling.md) \
Implementación adaptada de:  https://github.com/monologg/JointBERT

## Objetivo

El objetivo es entrenar un modelo de clasificación conjunta de intención y palabras claves para ser utilizado por el chatbot del Banco Central de Chile (PIBot). 

La clasificación está separada en dos partes, entender la intención de la consulta, ¿se pregunta por valores o metodología? e, identificar las palabras claves de la consulta: la métrica (PIB o IMACEC), el periodo, tipo de agregación, entre otros.

## Dataset

El principal objetivo de este proyecto es entrenar el modelo Joint BERT con un dataset especializado para detectar la intención y las palabras claves para consultas sobre PIB e IMACEC.

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
- inflación (si lo agregas)

#### 2. Periodo / fecha (el "cuándo")

Para meses, años, trimestres, fechas específicas:

- B-period
- I-period

Ejemplos:

- en septiembre de 2025
- en 2024
- en el tercer trimestre
- en los últimos doce meses
- ayer, hoy, último mes reportado

#### 3. Unidad / Métrica solicitada (el “cómo lo quieres”)

- B-measure_type
- I-measure_type

Ejemplos:

- variación anual / mensual / trimestral
- variación mensual desestacionalizada
- constribución / aporte / incidencia
- defactlor
- serie original
- crecimiento

#### 4. Actividad / Sector Económico (si la pregunta lo requiere)

Para capturar consultas específicas sobre sectores:

Slot:
- B-sector
- I-sector

Ejemplos:
- sector minero
- servicios personales
- producción de bienes
- comercio minorista

##### 5. Territorio / Región (si quisieras preguntas regionales)

Si lo incluyes:
- B-region
- I-region

Ejemplos:
- Región Metropolitana, Chile, Valparaíso

##### 6. Comparación (opcional)

Si quieres manejar:
- “comparado con el mes anterior”
- “respecto a junio”

Slot:
- B-comparison_period
- I-comparison_period

## Training & Evaluation

```
python main.py --task {task_name} \
                  --model_type {model_type} \
                  --model_out {model_out_name} \
                  --do_train --do_eval \
                  --use_crf
```
Ejemplo:
```
python main.py --task pibimacec --model_type beto --model_out pibot_model_beto --do_train --do_eval
```

## Prediction

```
python predict.py --input_file {INPUT_FILE_PATH} --output_file {OUTPUT_FILE_PATH} --model_dir {SAVED_CKPT_PATH}
```

Ejemplo:

```
python predict.py --input_file data/atis/test/seq.in --output_file output/atis.txt --model_dir model_out/atis_model
```

## Resultados

## Dependencias

- python>=3.12

## TODO

- Adaptar el train para modelos BERT en español como BETO
- Crear datasets de preguntas con intención y slots
- Entrenar el modelo y realizar las pruebas