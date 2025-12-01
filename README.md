# Clasificador de intención y palabras claves para PIBot usando Joint BERT

Paper DOI: [BERT for Joint Intent Classification and Slot Filling](https://arxiv.org/abs/1810.04805v2) \
Resumen paper: [docs/BERT_for_Joint_Intent_Classification_and_Slot_Filling.md](docs/BERT_for_Joint_Intent_Classification_and_Slot_Filling.md) \
Implementación adaptada de:  https://github.com/monologg/JointBERT

## Objetivo

El objetivo es entrenar un modelo de clasificación conjunta de intención y palabras claves para ser utilizado por el chatbot del Banco Central de Chile (PIBot). 

La clasificación está separada en dos partes, entender la intención de la consulta, ¿se pregunta por valores o metodología? e, identificar las palabras claves de la consulta: la métrica (PIB o IMACEC), el periodo, tipo de agregación, entre otros.

## Dataset

## Training & Evaluation

```
$ python main.py --task {task_name} \
                  --model_type {model_type} \
                  --model_out {model_out_name} \
                  --do_train --do_eval \
                  --use_crf
```

## Prediction

## Resultados

## Dependencias

- python>=3.12

## TODO

- Adaptar el train para modelos BERT en español como BETO
- Crear datasets de preguntas con intención y slots
- Entrenar el modelo y realizar las pruebas