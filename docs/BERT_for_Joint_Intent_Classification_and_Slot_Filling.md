# Resumen para paper: BERT for Joint Intent Classification and Slot Filling

DOI: https://arxiv.org/abs/1810.04805v2

PDF: https://export.arxiv.org/pdf/1902.10909v1

## Abstract

- La clasificación de intención y llenado de slots son dos tareas esencial para entender el lenguaje natural.
- Se propone la clasificación conjunta en un modelo basados en BERT.
- Predecir `intent` y `slot` al mismo tiempo utilizando un modelo tipo BERT (=Joint model).

## Introducción

- Natural Language Understanding (NLU) es fundamental para el rendimiento de los sistemas de dialogo.
- Problemas de NLU incluye la clasificación de intención y llenado de slots.
- La clasificación de la intención se centra en predecir la intención de una consulta.
- En llenado de slots extrae los conceptos semanticos.
- Antes el estado del arte era GRU (Gated Recurrent Unit) y LSTM (Long Short-Term Memory), luego se utilizaron nuevos enfoques que incluían atención.
 
## Enfoque propuesto

### BERT

- El modelo de la arquitectura BERT es un codificador transformer bidireccional de multiples capas.
- La representación del input es una concatenación de WordPiece embeddings, embeddings posicionales y la segmentación del embedding.
- Un clasificación especial de embedding ([CLS]) es insertado como el primer token y un token especial ([SEP]) es agregado como el token final. Dado un input de secuencia de token $x = (x_1,...,x_T)$, el output de BERT es $H = (h_1,...,h_T)$.
- El modelo BERT es pre-entrenado con dos estrategias en textos sin clasificar a gran escala: modelo de lenguaje enmascarado y la predicción de la siguiente oración.
- El modelo BERT pre-entrenado proporciona una podera representación de oraciones dependientes del contexto y puede ser usado para varias tareas.

### Joint Intent Classification and Slot Filling

BERT puede ser facilmente extendido a un modelo de clasificación conjunta de intención y llenado de slots. Basado en el estado oculto del primer token especial ([CLS]), denotado como $h_1$, la intención es predicha como:

$$
y^i = softmax(W^i h_1 + b^i)
$$

Para llenar los slots, nos alimentamos de los estados finales de los otros tokens $h_2,...,h_T$ en una capa softmax para clasificar las etiquetas de los slots.

$$
y_n^s = softmax(W^s h_n + b^s), n \in 1...N
$$

El modelo conjunto de clasificación de intención y llenado de slots es formulado como:

$$
p(y^i, y^s|x) = p(y^i|x) \prod_{n=1}^{N} p(y_n^s|x)
$$

El objetivo de aprendizaje es maximizar la probabilidad condicional $p(y^i, y^s|x)$. El modelo se ajusta de extremo a extremo mediante la minimización de la pérdida de la entropia cruzada.

### Campos aleatorios condicionales

La predicción de etiquetas de los slots es dependiente de la predicción de palabras circundantes. Se ha demostrado que los modelos de predicción pueden mejorar el rendimiento de llenado de slots, tal como los campos aleatorios condicionales (CRF). Se mejora el etiquetado añadiendo una capa CRF.

## Detalles de entrenamiento

Se utilizó el modelo __English uncased BERT-Base__ cual tiene 12 capas, 768 nodos ocultos y 12 cabezas. Para el __fine-tuning__, los hiper-parámetros configurados son: número máximo de palabras es 50, el __batch size__ es 128, Adam es utilizado para la optimización con el estado inicial de aprendizale $5e-5$, el __dropout probability__ es 0.1. El máximo número de epocas es seleccionado de [1, 5, 10, 20, 30, 40].

## Resultados del Joint BERT

### Snpis dataset 
- Intent = 98.6
- Slot = 97.0
- Sent = 92.8 

### ATIS dataset
- Intent 97.5
- Slot = 96.1
- Sent = 88.2

## Para futuras implementaciones

### Pre-Trained Joint Model for Intent Classification and Slot Filling with Semantic Feature Fusion
https://www.mdpi.com/1424-8220/23/5/2848

### Task Conditioned BERT for Joint Intent Detection and Slot-Filling
https://link.springer.com/chapter/10.1007/978-3-031-49008-8_37

### Complex Question Decomposition Based on Causal Reinforcement Learning
https://www.mdpi.com/2073-8994/17/7/1022?utm_source=chatgpt.com

### Question Decomposition Tree for Answering Complex Questions over Knowledge Bases
https://arxiv.org/abs/2306.07597?utm_source=chatgpt.com

### SPM: A Split-Parsing Method for Joint Multi-Intent Detection and Slot Filling
https://aclanthology.org/2023.acl-industry.64/?utm_source=chatgpt.com
