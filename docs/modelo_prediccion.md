# Cómo el Modelo Joint BERT Genera Predicciones

## Arquitectura General

El modelo Joint BERT está diseñado para realizar **dos tareas simultáneamente**:
1. **Clasificación de Intención** (Intent Classification): Determina la intención general de la consulta
2. **Etiquetado de Slots** (Slot Filling): Identifica y etiqueta entidades específicas en cada palabra

## Flujo de Predicción

### 1. Preprocesamiento del Texto

```python
# Entrada del usuario
texto = "cual fue el imacec de agosto 2024"

# Tokenización a nivel de palabras
words = texto.split()
# ['cual', 'fue', 'el', 'imacec', 'de', 'agosto', '2024']
```

#### a) Tokenización con WordPiece
BERT utiliza tokenización WordPiece que divide palabras en subpalabras:

```python
tokens = []
word_ids = []  # Mapea cada token a su palabra original

for i, word in enumerate(words):
    word_tokens = tokenizer.tokenize(word)
    # Ejemplo: "imacec" → ["imacec"]
    #          "agosto" → ["agosto"]
    tokens.extend(word_tokens)
    word_ids.extend([i] * len(word_tokens))
```

#### b) Agregar Tokens Especiales
```python
# Agregar [CLS] al inicio y [SEP] al final
tokens = ['[CLS]'] + tokens + ['[SEP]']
word_ids = [-1] + word_ids + [-1]
```

#### c) Convertir a IDs y Padding
```python
# Convertir tokens a IDs numéricos
input_ids = tokenizer.convert_tokens_to_ids(tokens)

# Crear máscaras de atención (1 = token real, 0 = padding)
attention_mask = [1] * len(input_ids)

# IDs de tipo de token (0 para secuencia única)
token_type_ids = [0] * len(input_ids)

# Padding hasta max_seq_len (típicamente 50 o 64)
padding_length = max_seq_len - len(input_ids)
input_ids += [tokenizer.pad_token_id] * padding_length
attention_mask += [0] * padding_length
token_type_ids += [0] * padding_length
```

### 2. Pasada por el Modelo BERT

```python
# Convertir a tensores PyTorch
input_ids = torch.tensor([input_ids])
attention_mask = torch.tensor([attention_mask])
token_type_ids = torch.tensor([token_type_ids])

# Pasada por BERT base
outputs = bert(
    input_ids=input_ids,
    attention_mask=attention_mask,
    token_type_ids=token_type_ids
)

sequence_output = outputs[0]  # Shape: [batch_size, seq_len, hidden_size=768]
pooled_output = outputs[1]    # Shape: [batch_size, hidden_size=768]
```

**BERT produce dos salidas principales:**
- `sequence_output`: Representación contextual de **cada token** (768 dimensiones por token)
- `pooled_output`: Representación del **token [CLS]** procesada (resumen de toda la secuencia)

### 3. Clasificación de Intención

```python
# El clasificador de intención usa el pooled_output ([CLS])
class IntentClassifier(nn.Module):
    def __init__(self, input_dim=768, num_intent_labels, dropout_rate=0.1):
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_intent_labels)
    
    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)  # Shape: [batch_size, num_intents]

# Predicción
intent_logits = intent_classifier(pooled_output)
# Ejemplo: [batch_size, 2] → [[-0.5, 2.3]] para ['methodology', 'value']

# Aplicar softmax para obtener probabilidades
intent_probs = torch.softmax(intent_logits, dim=-1)
# [[0.06, 0.94]] → 94% de confianza en 'value'

# Obtener predicción
intent_pred = intent_logits.argmax(dim=-1).item()  # → 1
intent_label = intent_labels[intent_pred]  # → 'value'
confidence = intent_probs[0][intent_pred].item()  # → 0.94
```

### 4. Etiquetado de Slots (con CRF)

#### a) Clasificador de Slots
```python
class SlotClassifier(nn.Module):
    def __init__(self, input_dim=768, num_slot_labels, dropout_rate=0.1):
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_slot_labels)
    
    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)  # Shape: [batch_size, seq_len, num_slots]

# Predicción
slot_logits = slot_classifier(sequence_output)
# Shape: [1, seq_len, num_slots] → scores para cada etiqueta BIO en cada posición
```

#### b) Decodificación con CRF (Conditional Random Field)

El CRF mejora las predicciones considerando **transiciones válidas entre etiquetas**:

```python
# Sin CRF (independiente):
# Podría predecir: ['O', 'I-period', 'B-indicator', 'O']  ❌ INVÁLIDO
# (I-period no puede aparecer sin B-period antes)

# Con CRF:
# Aprende que I-X solo puede seguir a B-X o I-X
slot_preds = model.crf.decode(slot_logits, mask=attention_mask.bool())[0]
# → ['O', 'O', 'O', 'B-indicator', 'O', 'B-period', 'I-period']  ✓ VÁLIDO
```

**Restricciones del CRF:**
- `I-X` solo puede seguir a `B-X` o `I-X`
- `B-X` puede aparecer en cualquier momento
- `O` puede aparecer en cualquier momento

### 5. Mapeo de Tokens a Palabras

Debido a la tokenización WordPiece, múltiples tokens pueden corresponder a una palabra:

```python
# Ejemplo con palabra compuesta
word = "Banco-Central"
tokens = ['Banco', '-', 'Central']
word_ids = [5, 5, 5]  # Todos pertenecen a la palabra 5

# Para obtener el slot de cada palabra original:
slot_labels = []
for i, word in enumerate(words):
    # Encontrar el primer subtoken de esta palabra
    subtoken_idx = word_ids.index(i)
    slot_label = slot_labels_list[slot_preds[subtoken_idx]]
    slot_labels.append(slot_label)
```

### 6. Salida Final

```python
{
    'text': 'cual fue el imacec de agosto 2024',
    'words': ['cual', 'fue', 'el', 'imacec', 'de', 'agosto', '2024'],
    'intent': 'value',
    'intent_confidence': 0.94,
    'slots': ['O', 'O', 'O', 'B-indicator', 'O', 'B-period', 'I-period']
}
```

## Componentes del Modelo

### Estructura de `JointBERT`

```
Input Text
    ↓
[Tokenización WordPiece]
    ↓
BERT Encoder (12 capas)
    ├── sequence_output (todos los tokens) → SlotClassifier → CRF → Slot Labels
    └── pooled_output ([CLS] token)       → IntentClassifier → Intent + Confidence
```

### Parámetros Entrenables

1. **BERT Base**: ~110M parámetros (pre-entrenado, fine-tuned)
2. **Intent Classifier**: 768 × num_intents (ej: 768 × 2 = 1,536 parámetros)
3. **Slot Classifier**: 768 × num_slots (ej: 768 × 9 = 6,912 parámetros)
4. **CRF**: num_slots² (para transiciones) (ej: 9² = 81 parámetros)

**Total**: ~110M parámetros

## Ventajas del Enfoque Joint

### 1. Aprendizaje Conjunto
- La intención y los slots se aprenden **simultáneamente**
- Compartir representaciones BERT mejora ambas tareas
- La pérdida combinada: `loss = intent_loss + slot_loss_coef * slot_loss`

### 2. Contexto Bidireccional
- BERT procesa la secuencia completa en ambas direcciones
- Cada palabra tiene contexto de las palabras anteriores y posteriores
- Ejemplo: "banco" puede ser institución financiera o asiento según el contexto

### 3. CRF para Coherencia
- Garantiza secuencias válidas de etiquetas BIO
- Evita transiciones imposibles (ej: I-period sin B-period previo)
- Aprende patrones de transición del dataset de entrenamiento

## Ejemplo Completo: Traza de Predicción

```python
# INPUT
"cuanto subio el imacec del mes pasado"

# TOKENIZACIÓN
['[CLS]', 'cuanto', 'subio', 'el', 'imacec', 'del', 'mes', 'pasado', '[SEP]']
word_ids: [-1, 0, 1, 2, 3, 4, 5, 6, -1]

# BERT ENCODER
sequence_output.shape = [1, 9, 768]  # 9 tokens, 768 dimensiones cada uno
pooled_output.shape = [1, 768]       # Representación [CLS]

# INTENT CLASSIFICATION
intent_logits = [-1.2, 2.8]  # [methodology, value]
softmax → [0.02, 0.98]
→ Intent: 'value', Confidence: 98%

# SLOT CLASSIFICATION + CRF
slot_logits.shape = [1, 9, 9]  # 9 tokens, 9 posibles etiquetas
CRF decode → [0, 0, 0, 1, 0, 2, 3, 0]  # IDs de etiquetas
→ Slots: ['O', 'O', 'O', 'B-indicator', 'O', 'B-period', 'I-period', 'O']

# OUTPUT (mapeado a palabras originales, excluyendo [CLS] y [SEP])
{
    'intent': 'value',
    'confidence': 0.98,
    'entities': {
        'indicator': 'imacec',
        'period': 'mes pasado'
    }
}
```

## Optimizaciones Durante Inferencia

1. **`model.eval()`**: Desactiva dropout para predicciones consistentes
2. **`torch.no_grad()`**: No calcula gradientes (más rápido, menos memoria)
3. **Batch processing**: Puede procesar múltiples textos simultáneamente
4. **GPU/CPU**: Detecta automáticamente el dispositivo disponible

## Limitaciones

1. **Max Sequence Length**: Textos muy largos se truncan (típicamente 50-64 tokens)
2. **Vocabulario OOV**: Palabras desconocidas se mapean a `[UNK]`
3. **Dependencia de datos**: La calidad depende del dataset de entrenamiento
4. **Latencia**: BERT es relativamente lento comparado con modelos más simples
