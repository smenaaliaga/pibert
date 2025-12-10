"""
Predicción interactiva desde línea de comandos.

Uso:
    # Modo interactivo
    python predict_cli.py --model_dir model_out/pibot_model_beto
    
    # Texto único
    python predict_cli.py --model_dir model_out/pibot_model_beto --text "cual fue el ultimo imacec"
    
    # Batch de textos
    python predict_cli.py --model_dir model_out/pibot_model_beto --input_file consultas.txt --output_file resultados.txt
"""

import argparse
import os
import logging
import torch
from transformers import BertTokenizer

from utils import get_intent_labels, get_slot_labels, MODEL_CLASSES, init_logger

logger = logging.getLogger(__name__)


class Predictor:
    def __init__(self, model_dir, no_cuda=False):
        self.model_dir = model_dir
        self.device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"
        
        # Cargar args del entrenamiento
        self.args = torch.load(os.path.join(model_dir, 'training_args.bin'), weights_only=False)
        
        # Cargar labels
        self.intent_label_lst = get_intent_labels(self.args)
        self.slot_label_lst = get_slot_labels(self.args)
        
        # Cargar tokenizer
        self.tokenizer = MODEL_CLASSES[self.args.model_type][2].from_pretrained(self.args.model_name_or_path)
        
        # Cargar modelo
        config_class, model_class, _ = MODEL_CLASSES[self.args.model_type]
        self.model = model_class.from_pretrained(
            model_dir,
            config=config_class.from_pretrained(model_dir),
            args=self.args,
            intent_label_lst=self.intent_label_lst,
            slot_label_lst=self.slot_label_lst
        )
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded from {model_dir}")
        logger.info(f"Device: {self.device}")
    
    def predict(self, text):
        """Predice intención y slots para un texto."""
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
        max_seq_len = self.args.max_seq_len
        if len(tokens) > max_seq_len - 2:
            tokens = tokens[:(max_seq_len - 2)]
            word_ids = word_ids[:(max_seq_len - 2)]
        
        # Agregar tokens especiales
        tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
        word_ids = [-1] + word_ids + [-1]
        
        # Convertir a IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)
        
        # Padding
        padding_length = max_seq_len - len(input_ids)
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
                slot_preds = self.model.crf.decode(slot_logits, mask=attention_mask.bool())[0]
            else:
                slot_preds = slot_logits.argmax(dim=-1)[0].cpu().numpy()
            
            # Mapear slots a palabras originales
            slot_labels = []
            for i, word in enumerate(words):
                # Encontrar el primer subtoken de esta palabra
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
    
    def format_result(self, result):
        """Formatea el resultado para mostrar en consola."""
        output = []
        output.append(f"\n{'='*60}")
        output.append(f"Texto: {result['text']}")
        output.append(f"{'='*60}")
        output.append(f"\nIntención: {result['intent']} (confianza: {result['intent_confidence']:.2%})")
        output.append(f"\nSlots detectados:")
        output.append(f"{'Palabra':<20} {'Etiqueta':<20}")
        output.append(f"{'-'*40}")
        
        for word, slot in zip(result['words'], result['slots']):
            if slot != 'O':
                output.append(f"{word:<20} {slot:<20}")
        
        if all(s == 'O' for s in result['slots']):
            output.append("  (ningún slot detectado)")
        
        output.append(f"{'='*60}\n")
        return '\n'.join(output)


def interactive_mode(predictor):
    """Modo interactivo."""
    print("\n" + "="*60)
    print("MODO INTERACTIVO - PIBot Joint BERT")
    print("="*60)
    print("Escribe una consulta y presiona Enter.")
    print("Escribe 'salir' o 'exit' para terminar.\n")
    
    while True:
        try:
            text = input("Consulta: ").strip()
            
            if not text:
                continue
            
            if text.lower() in ['salir', 'exit', 'quit', 'q']:
                print("¡Hasta luego!")
                break
            
            result = predictor.predict(text)
            print(predictor.format_result(result))
            
        except KeyboardInterrupt:
            print("\n¡Hasta luego!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"Error al procesar: {e}\n")


def batch_mode(predictor, input_file, output_file=None):
    """Modo batch desde archivo."""
    with open(input_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    results = []
    print(f"\nProcesando {len(texts)} consultas...")
    
    for i, text in enumerate(texts, 1):
        try:
            result = predictor.predict(text)
            results.append(result)
            print(f"[{i}/{len(texts)}] ✓ {text[:50]}...")
        except Exception as e:
            logger.error(f"Error en línea {i}: {e}")
            print(f"[{i}/{len(texts)}] ✗ Error: {e}")
    
    # Mostrar resultados
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(predictor.format_result(result))
                f.write("\n")
        print(f"\n✓ Resultados guardados en: {output_file}")
    else:
        for result in results:
            print(predictor.format_result(result))


def main():
    init_logger()
    
    parser = argparse.ArgumentParser(description='Predicción CLI para Joint BERT')
    parser.add_argument('--model_dir', type=str, required=True, help='Directorio del modelo entrenado')
    parser.add_argument('--text', type=str, help='Texto único a predecir')
    parser.add_argument('--input_file', type=str, help='Archivo con textos (uno por línea)')
    parser.add_argument('--output_file', type=str, help='Archivo de salida para resultados')
    parser.add_argument('--no_cuda', action='store_true', help='No usar CUDA')
    
    args = parser.parse_args()
    
    # Cargar predictor
    try:
        predictor = Predictor(args.model_dir, args.no_cuda)
    except Exception as e:
        logger.error(f"Error cargando modelo: {e}")
        print(f"Error: No se pudo cargar el modelo desde {args.model_dir}")
        print(f"Detalle: {e}")
        return
    
    # Modo de operación
    if args.text:
        # Texto único
        result = predictor.predict(args.text)
        print(predictor.format_result(result))
    elif args.input_file:
        # Batch
        batch_mode(predictor, args.input_file, args.output_file)
    else:
        # Interactivo
        interactive_mode(predictor)


if __name__ == '__main__':
    main()
