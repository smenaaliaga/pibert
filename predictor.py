"""
Módulo de predicción completo para PIBot Joint BERT.

Este módulo proporciona una interfaz simple para realizar predicciones
de intención y extracción de entidades, ideal para ser usado por agentes
o APIs.

Uso:
    from predictor import PIBotPredictor
    
    # Inicializar
    predictor = PIBotPredictor(model_dir='model_out/pibot_model_beto')
    
    # Predecir
    result = predictor.predict("cual fue el imacec de agosto 2024")
    
    # Resultado:
    # {
    #     'intent': 'value',
    #     'confidence': 0.98,
    #     'entities': {
    #         'indicator': 'imacec',
    #         'period': 'agosto 2024'
    #     }
    # }
"""

import os
import logging
import torch
from datetime import date
from typing import Dict, Optional, Any
from transformers import BertTokenizer

from utils import get_intent_labels, get_slot_labels, MODEL_CLASSES
from util.period_normalizer import standardize_imacec_time_ref
from util.indicator_normalizer import standardize_indicator

logger = logging.getLogger(__name__)


class PIBotPredictor:
    """
    Predictor completo para PIBot que encapsula toda la lógica de predicción.
    
    Este predictor:
    1. Carga el modelo entrenado
    2. Procesa el texto de entrada
    3. Extrae intención y entidades
    4. Normaliza entidades (fechas e indicadores)
    5. Retorna un formato estructurado y limpio
    """
    
    def __init__(
        self,
        model_dir: str,
        device: Optional[str] = None,
        normalize_entities: bool = True,
        max_seq_len: int = 50
    ):
        """
        Inicializa el predictor.
        
        Args:
            model_dir: Directorio con el modelo entrenado
            device: 'cuda', 'cpu', o None (auto-detectar)
            normalize_entities: Si True, normaliza fechas e indicadores
            max_seq_len: Longitud máxima de secuencia
        """
        self.model_dir = model_dir
        self.normalize_entities = normalize_entities
        self.max_seq_len = max_seq_len
        
        # Detectar dispositivo
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Cargar modelo
        self._load_model()
        
        logger.info(f"PIBotPredictor inicializado en {self.device}")
    
    def _load_model(self):
        """Carga el modelo y componentes necesarios."""
        # Cargar args
        args_path = os.path.join(self.model_dir, 'training_args.bin')
        if not os.path.exists(args_path):
            raise FileNotFoundError(f"No se encuentra {args_path}")
        
        self.args = torch.load(args_path)
        self.args.max_seq_len = self.max_seq_len  # Override si es necesario
        
        # Cargar labels
        self.intent_label_lst = get_intent_labels(self.args)
        self.slot_label_lst = get_slot_labels(self.args)
        
        # Cargar tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.args.model_name_or_path)
        
        # Cargar modelo
        model_class = MODEL_CLASSES[self.args.model_type][1]
        self.model = model_class.from_pretrained(
            self.model_dir,
            args=self.args,
            intent_label_lst=self.intent_label_lst,
            slot_label_lst=self.slot_label_lst
        )
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Modelo cargado: {len(self.intent_label_lst)} intenciones, "
                   f"{len(self.slot_label_lst)} slots")
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Realiza predicción completa sobre el texto.
        
        Args:
            text: Texto de entrada (consulta del usuario)
        
        Returns:
            Dict con formato:
            {
                'intent': str,              # Intención detectada
                'confidence': float,        # Confianza 0-1
                'entities': {               # Entidades extraídas
                    'indicator': str,       # Opcional
                    'period': str,          # Opcional
                    'measure_type': str,    # Opcional
                    ...
                },
                'normalized': {             # Opcional: entidades normalizadas
                    'indicator': {...},
                    'period': {...}
                }
            }
        """
        if not text or not text.strip():
            return {
                'intent': None,
                'confidence': 0.0,
                'entities': {}
            }
        
        # 1. Predicción básica (intent + slots)
        raw_prediction = self._predict_raw(text)
        
        # 2. Extraer entidades de los slots
        entities = self._extract_entities(
            raw_prediction['words'],
            raw_prediction['slots']
        )
        
        # 3. Normalizar entidades si está habilitado
        normalized = {}
        if self.normalize_entities:
            normalized = self._normalize_entities(entities)
        
        # 4. Construir respuesta
        result = {
            'intent': raw_prediction['intent'],
            'confidence': raw_prediction['intent_confidence'],
            'entities': entities
        }
        
        if normalized:
            result['normalized'] = normalized
        
        return result
    
    def _predict_raw(self, text: str) -> Dict[str, Any]:
        """
        Predicción básica del modelo (sin post-procesamiento).
        
        Returns:
            Dict con intent, confidence, words, slots
        """
        # Tokenizar
        words = text.split()
        tokens = []
        word_ids = []
        
        for i, word in enumerate(words):
            word_tokens = self.tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [self.tokenizer.unk_token]
            tokens.extend(word_tokens)
            word_ids.extend([i] * len(word_tokens))
        
        # Truncar si es necesario
        if len(tokens) > self.args.max_seq_len - 2:
            tokens = tokens[:(self.args.max_seq_len - 2)]
            word_ids = word_ids[:(self.args.max_seq_len - 2)]
        
        # Agregar tokens especiales
        tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
        word_ids = [-1] + word_ids + [-1]
        
        # Convertir a IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)
        
        # Padding
        padding_length = self.args.max_seq_len - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        token_type_ids += [0] * padding_length
        
        # Convertir a tensors
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(self.device)
        token_type_ids = torch.tensor([token_type_ids], dtype=torch.long).to(self.device)
        
        # Predecir
        with torch.no_grad():
            inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'intent_label_ids': None,
                'slot_labels_ids': None
            }
            outputs = self.model(**inputs)
            _, (intent_logits, slot_logits) = outputs[:2]
            
            # Intent
            intent_pred = intent_logits.argmax(dim=-1).item()
            intent_label = self.intent_label_lst[intent_pred]
            intent_confidence = torch.softmax(intent_logits, dim=-1)[0][intent_pred].item()
            
            # Slots
            if self.args.use_crf:
                slot_preds = self.model.crf.decode(slot_logits)[0]
            else:
                slot_preds = slot_logits.argmax(dim=-1)[0].cpu().numpy()
            
            # Mapear slots a palabras originales
            slot_labels = []
            for i, word in enumerate(words):
                subtoken_idx = word_ids.index(i) if i in word_ids else -1
                if subtoken_idx >= 0 and subtoken_idx < len(slot_preds):
                    slot_labels.append(self.slot_label_lst[slot_preds[subtoken_idx]])
                else:
                    slot_labels.append("O")
        
        return {
            'text': text,
            'words': words,
            'intent': intent_label,
            'intent_confidence': intent_confidence,
            'slots': slot_labels
        }
    
    def _extract_entities(self, words: list, slots: list) -> Dict[str, str]:
        """
        Extrae entidades de las etiquetas BIO.
        
        Args:
            words: Lista de palabras
            slots: Lista de etiquetas BIO
        
        Returns:
            Dict con entidades extraídas {'indicator': 'imacec', 'period': 'agosto 2024'}
        """
        entities = {}
        current_entity = None
        current_value = []
        
        for word, slot in zip(words, slots):
            if slot.startswith('B-'):
                # Guardar entidad anterior si existe
                if current_entity:
                    entities[current_entity] = ' '.join(current_value)
                # Iniciar nueva entidad
                current_entity = slot[2:]  # Remover 'B-'
                current_value = [word]
            elif slot.startswith('I-') and current_entity:
                # Continuar entidad actual
                current_value.append(word)
            else:
                # Finalizar entidad actual si existe
                if current_entity:
                    entities[current_entity] = ' '.join(current_value)
                    current_entity = None
                    current_value = []
        
        # Guardar última entidad si existe
        if current_entity:
            entities[current_entity] = ' '.join(current_value)
        
        return entities
    
    def _normalize_entities(self, entities: Dict[str, str]) -> Dict[str, Any]:
        """
        Normaliza las entidades extraídas.
        
        Args:
            entities: Dict con entidades sin normalizar
        
        Returns:
            Dict con entidades normalizadas
        """
        normalized = {}
        
        # Normalizar período
        if 'period' in entities:
            period_normalized = standardize_imacec_time_ref(
                entities['period'],
                date.today()
            )
            if period_normalized:
                normalized['period'] = period_normalized
        
        # Normalizar indicador
        if 'indicator' in entities:
            indicator_normalized = standardize_indicator(entities['indicator'])
            if indicator_normalized and indicator_normalized.get('indicator'):
                normalized['indicator'] = {
                    'standard_name': indicator_normalized['indicator'].upper(),
                    'text_normalized': indicator_normalized['text_norm'],
                    'detected_by': indicator_normalized.get('text_standardized_imacec')
                }
        
        return normalized
    
    def predict_batch(self, texts: list) -> list:
        """
        Predice múltiples textos de forma secuencial.
        
        Args:
            texts: Lista de textos
        
        Returns:
            Lista de predicciones
        """
        return [self.predict(text) for text in texts]
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Retorna información sobre el modelo cargado.
        
        Returns:
            Dict con información del modelo
        """
        return {
            'model_dir': self.model_dir,
            'model_type': self.args.model_type,
            'device': self.device,
            'num_intents': len(self.intent_label_lst),
            'intent_labels': self.intent_label_lst,
            'num_slots': len(self.slot_label_lst),
            'slot_labels': self.slot_label_lst,
            'max_seq_len': self.args.max_seq_len,
            'use_crf': self.args.use_crf,
            'normalize_entities': self.normalize_entities
        }


# Función de conveniencia para uso rápido
_global_predictor = None

def get_predictor(model_dir: str = 'model_out/pibot_model_beto', **kwargs) -> PIBotPredictor:
    """
    Obtiene una instancia global del predictor (patrón singleton).
    
    Args:
        model_dir: Directorio del modelo
        **kwargs: Argumentos adicionales para PIBotPredictor
    
    Returns:
        Instancia de PIBotPredictor
    """
    global _global_predictor
    if _global_predictor is None:
        _global_predictor = PIBotPredictor(model_dir, **kwargs)
    return _global_predictor


def predict(text: str, model_dir: str = 'model_out/pibot_model_beto') -> Dict[str, Any]:
    """
    Función de conveniencia para predicción rápida.
    
    Args:
        text: Texto a predecir
        model_dir: Directorio del modelo
    
    Returns:
        Dict con predicción
    """
    predictor = get_predictor(model_dir)
    return predictor.predict(text)


# Ejemplo de uso
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Inicializar predictor
    print("Inicializando predictor...")
    predictor = PIBotPredictor(
        model_dir='model_out/pibot_model_beto',
        normalize_entities=True
    )
    
    # Información del modelo
    print("\n" + "="*70)
    print("INFORMACIÓN DEL MODELO")
    print("="*70)
    info = predictor.get_model_info()
    print(f"Tipo de modelo: {info['model_type']}")
    print(f"Dispositivo: {info['device']}")
    print(f"Intenciones: {info['intent_labels']}")
    print(f"Número de slots: {info['num_slots']}")
    print(f"CRF activado: {info['use_crf']}")
    
    # Ejemplos de predicción
    print("\n" + "="*70)
    print("EJEMPLOS DE PREDICCIÓN")
    print("="*70 + "\n")
    
    test_queries = [
        "cual fue el imacec de agosto 2024",
        "dame el pib del ultimo trimestre",
        "como estuvo el imacec el mes pasado",
        "cuanto subio la mineria en el imacec",
        "quiero saber el imacec mas reciente"
    ]
    
    for query in test_queries:
        print(f"Consulta: '{query}'")
        result = predictor.predict(query)
        
        print(f"  Intent: {result['intent']} (confianza: {result['confidence']:.2%})")
        
        if result['entities']:
            print(f"  Entidades:")
            for entity_type, entity_value in result['entities'].items():
                print(f"    - {entity_type}: {entity_value}")
        
        if 'normalized' in result and result['normalized']:
            print(f"  Normalizadas:")
            if 'period' in result['normalized']:
                period = result['normalized']['period']
                print(f"    - period: {period['period_key']} "
                      f"(granularidad: {period['granularity']}, "
                      f"confianza: {period['confidence']})")
            if 'indicator' in result['normalized']:
                indicator = result['normalized']['indicator']
                print(f"    - indicator: {indicator['standard_name']}")
        
        print()
    
    # Ejemplo de predicción en batch
    print("="*70)
    print("PREDICCIÓN EN BATCH")
    print("="*70 + "\n")
    
    batch_results = predictor.predict_batch(test_queries[:3])
    for i, result in enumerate(batch_results, 1):
        print(f"{i}. {result['intent']} ({result['confidence']:.2%}) - "
              f"{len(result['entities'])} entidades")
