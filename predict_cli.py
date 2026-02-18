"""
Predicción interactiva desde línea de comandos.

Uso:
    # Modo interactivo
    python predict_cli.py --model_dir model_out/pibot_model_v5_beto_crf
    
    # Texto único
    python predict_cli.py --model_dir model_out/pibot_model_v5_beto_crf --text "cual fue el ultimo imacec"
    
    # Batch de textos
    python predict_cli.py --model_dir model_out/pibot_model_v5_beto_crf --input_file consultas.txt --output_file resultados.txt
"""

import argparse
import os
import logging
from collections import defaultdict
import torch

from utils import (
    get_calc_mode_labels,
    get_activity_labels,
    get_region_labels,
    get_investment_labels,
    get_req_form_labels,
    get_slot_labels,
    MODEL_CONFIG,
    init_logger,
)

logger = logging.getLogger(__name__)


class Predictor:
    def __init__(self, model_dir, no_cuda=False):
        self.model_dir = model_dir
        # Detectar dispositivo: CUDA > MPS > CPU
        if torch.cuda.is_available() and not no_cuda:
            self.device = "cuda"
        elif torch.backends.mps.is_available() and not no_cuda:
            self.device = "mps"
        else:
            self.device = "cpu"
        
        # Cargar args del entrenamiento
        self.args = torch.load(os.path.join(model_dir, 'training_args.bin'), weights_only=False)
        
        # Cargar labels (5 cabezas)
        self.calc_mode_label_lst = get_calc_mode_labels(self.args)
        self.activity_label_lst = get_activity_labels(self.args)
        self.region_label_lst = get_region_labels(self.args)
        self.investment_label_lst = get_investment_labels(self.args)
        self.req_form_label_lst = get_req_form_labels(self.args)
        self.slot_label_lst = get_slot_labels(self.args)
        
        # Cargar tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path)
        
        # Cargar modelo
        config_class, model_class, _ = MODEL_CONFIG
        self.model = model_class.from_pretrained(
            model_dir,
            config=config_class.from_pretrained(model_dir),
            args=self.args,
            calc_mode_label_lst=self.calc_mode_label_lst,
            activity_label_lst=self.activity_label_lst,
            region_label_lst=self.region_label_lst,
            investment_label_lst=self.investment_label_lst,
            req_form_label_lst=self.req_form_label_lst,
            slot_label_lst=self.slot_label_lst
        )
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded from {model_dir}")
        logger.info(f"Device: {self.device}")

    @staticmethod
    def _extract_entities_from_bio(words, bio_tags):
        """Convierte etiquetas BIO por palabra a entidades agrupadas."""
        entities = []
        current_type = None
        current_tokens = []

        for word, tag in zip(words, bio_tags):
            if not tag or tag == "O":
                if current_type and current_tokens:
                    entities.append((current_type, " ".join(current_tokens)))
                current_type = None
                current_tokens = []
                continue

            if "-" not in tag:
                if current_type and current_tokens:
                    entities.append((current_type, " ".join(current_tokens)))
                current_type = None
                current_tokens = []
                continue

            prefix, entity_type = tag.split("-", 1)

            if prefix == "B":
                if current_type and current_tokens:
                    entities.append((current_type, " ".join(current_tokens)))
                current_type = entity_type
                current_tokens = [word]
            elif prefix == "I" and current_type == entity_type:
                current_tokens.append(word)
            else:
                if current_type and current_tokens:
                    entities.append((current_type, " ".join(current_tokens)))
                current_type = entity_type
                current_tokens = [word]

        if current_type and current_tokens:
            entities.append((current_type, " ".join(current_tokens)))

        grouped = defaultdict(list)
        for entity_type, text in entities:
            grouped[entity_type].append(text)
        return dict(grouped)
    
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
                'calc_mode_label_ids': None,
                'activity_label_ids': None,
                'region_label_ids': None,
                'investment_label_ids': None,
                'req_form_label_ids': None,
                'slot_labels_ids': None
            }
            outputs = self.model(**inputs)
            logits = outputs[1]
            (
                calc_mode_logits,
                activity_logits,
                region_logits,
                investment_logits,
                req_form_logits,
                slot_logits,
            ) = logits
            
            def predict_head(logit_tensor, label_list):
                """Return (label, confidence) for a classification head."""
                if logit_tensor is None or logit_tensor.numel() == 0:
                    return None, None
                pred_idx = logit_tensor.argmax(dim=-1).item()
                label = label_list[pred_idx] if label_list else None
                confidence = torch.softmax(logit_tensor, dim=-1)[0][pred_idx].item()
                return label, confidence

            calc_mode_label, calc_mode_confidence = predict_head(calc_mode_logits, self.calc_mode_label_lst)
            activity_label, activity_confidence = predict_head(activity_logits, self.activity_label_lst)
            region_label, region_confidence = predict_head(region_logits, self.region_label_lst)
            investment_label, investment_confidence = predict_head(investment_logits, self.investment_label_lst)
            req_form_label, req_form_confidence = predict_head(req_form_logits, self.req_form_label_lst)

            # Decodificar NER (slot) por palabra
            if getattr(self.args, 'use_crf', False) and hasattr(self.model, 'crf'):
                if hasattr(self.model.crf, 'decode'):
                    slot_pred_ids = self.model.crf.decode(slot_logits, mask=attention_mask.bool())[0]
                else:
                    slot_pred_ids = self.model.crf.viterbi_decode(slot_logits, attention_mask.bool())[0]
            else:
                slot_pred_ids = torch.argmax(slot_logits, dim=-1).squeeze(0).tolist()

            slot_tags_per_word = ["O"] * len(words)
            seen_word_idxs = set()
            for idx, word_idx in enumerate(word_ids):
                if word_idx < 0 or word_idx in seen_word_idxs:
                    continue
                seen_word_idxs.add(word_idx)
                if idx < len(slot_pred_ids):
                    pred_id = slot_pred_ids[idx]
                    if 0 <= pred_id < len(self.slot_label_lst):
                        slot_tags_per_word[word_idx] = self.slot_label_lst[pred_id]
                    else:
                        slot_tags_per_word[word_idx] = "O"

            entities = self._extract_entities_from_bio(words, slot_tags_per_word)
        
        return {
            'text': text,
            'words': words,
            'calc_mode': calc_mode_label,
            'calc_mode_confidence': calc_mode_confidence,
            'activity': activity_label,
            'activity_confidence': activity_confidence,
            'region': region_label,
            'region_confidence': region_confidence,
            'investment': investment_label,
            'investment_confidence': investment_confidence,
            'req_form': req_form_label,
            'req_form_confidence': req_form_confidence,
            'slot_tags': slot_tags_per_word,
            'entities': entities,
        }
    
    def format_result(self, result):
        """Formatea el resultado para mostrar en consola."""
        output = []
        output.append(f"\n{'='*60}")
        # output.append(f"text: {result['text']}")
        # output.append(f"{'='*60}")
        output.append(f"calc_mode\t: {result.get('calc_mode') or 'N/A':<20}\t(confidence: {result.get('calc_mode_confidence', 0):.2%})" if result.get('calc_mode_confidence') is not None else f"calc_mode\t: {result.get('calc_mode') or 'N/A'}")
        output.append(f"activity\t: {result.get('activity') or 'N/A':<20}\t(confidence: {result.get('activity_confidence', 0):.2%})" if result.get('activity_confidence') is not None else f"activity\t: {result.get('activity') or 'N/A'}")
        output.append(f"region\t\t: {result.get('region') or 'N/A':<20}\t(confidence: {result.get('region_confidence', 0):.2%})" if result.get('region_confidence') is not None else f"region\t\t: {result.get('region') or 'N/A'}")
        output.append(f"investment\t: {result.get('investment') or 'N/A':<20}\t(confidence: {result.get('investment_confidence', 0):.2%})" if result.get('investment_confidence') is not None else f"investment\t: {result.get('investment') or 'N/A'}")
        output.append(f"req_form\t: {result.get('req_form') or 'N/A':<20}\t(confidence: {result.get('req_form_confidence', 0):.2%})" if result.get('req_form_confidence') is not None else f"req_form\t: {result.get('req_form') or 'N/A'}")
        output.append(f"{'='*60}")

        entities = result.get('entities') or {}
        if entities:
            output.append("Entidades detectadas:")
            for entity_type, mentions in entities.items():
                output.append(f"  - {entity_type}: [{', '.join(mentions)}]")
        else:
            output.append("Entidades detectadas: []")

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
