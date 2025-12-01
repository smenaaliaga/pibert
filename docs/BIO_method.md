# Método BIO

El esquema BIO es la forma más común de marcar los slots palabra por palabra en tareas de _slot filling_.

BIO viene de:

- B = Beging.
- I = Inside.
- O = Outside.

La idea es que tu modelo no predice qué tipo de slot es cada palabra, sino también si es el inicio o la continuación de ese slot.

## ¿Qué es el slot filling?

En _spoken language understanding_, podemos tener la frase (ATIS):

```
quiero volar de santiago a nueva york mañana
```

Suponiendo los siguientes slots:

- santiago → from_city
- nueva york → to_city
- mañana → date

Con BIO, etiquetas cada token así:

| Token     | Tag BIO          |
|-----------|------------------|
| quiero    | O                |
| volar     | O                |
| de        | O                |
| santiago  | B-from_city      |
| a         | O                |
| nueva     | B-to_city        |
| york      | I-to_city        |
| mañana    | B-date           |

Explicación:

- B-from_city: comienzo del slot from_city.
- I-to_city: continuación del slot to_city.
- O: token que no pertenece a ningún slot.

## ¿Cómo se ve en el modelo?

### En entrenamiento

- Entrada: la secuencia de tokens (a veces subwords si usas BERT).
- Salida: una secuencia de etiquetas BIO, por ejemplo:

```
["O", "O", "O", "B-from_city", "O", "B-to_city", "I-to_city", "B-date"]
```

El modelo aprende una clasificación por token donde el conjunto de etiquetas podría ser, por ejemplo:

```
["O", "O", "O", "B-from_city", "O", "B-to_city", "I-to_city", "B-date"]
```

### Inferencia

El modelo predice una etiqueta BIO por token.