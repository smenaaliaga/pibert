"""
Predicción interactiva desde línea de comandos.

Uso:
    # Modo interactivo
    python predict_cli.py --model_dir model_out/pibot_model_beto_v2
    
    # Texto único
    python predict_cli.py --model_dir model_out/pibot_model_beto --text "cual fue el ultimo imacec"
    
    # Batch de textos
    python predict_cli.py --model_dir model_out/pibot_model_beto --input_file consultas.txt --output_file resultados.txt
"""

import argparse
import os
import logging
import sys
import json
from pathlib import Path
from datetime import date
import torch
from transformers import BertTokenizer

from utils import (
    get_intent_labels,
    get_indicator_labels,
    get_metric_type_labels,
    get_calc_mode_labels,
    get_seasonal_labels,
    get_req_form_labels,
    get_context_mode_labels,
    get_activity_labels,
    get_slot_labels,
    MODEL_CLASSES,
    init_logger,
)

# Cargar normalizador de fechas desde util/normalizer_date.py
_util_path = Path(__file__).resolve().parent / "util"
if not _util_path.exists():
    _util_path = Path(__file__).resolve().parent.parent / "util"
sys.path.append(str(_util_path))
from normalizer_date import extract_month_period, extract_month_range  # noqa: E402

_SERIES_PATH = Path(__file__).resolve().parent / "series_catalog.json"


def _load_series_catalog():
    if not hasattr(_load_series_catalog, "_cache"):
        try:
            with open(_SERIES_PATH, "r", encoding="utf-8") as f:
                _load_series_catalog._cache = json.load(f)
        except Exception:
            _load_series_catalog._cache = {}
    return _load_series_catalog._cache


def _find_series(indicator: str, activity: str, seasonal: str, metric_type: str):
    catalog = _load_series_catalog()
    indicator_l = indicator.lower() if indicator else None
    activity_l = activity.lower() if activity else None
    seasonal_l = seasonal.lower() if seasonal else None
    metric_type_l = metric_type.lower() if metric_type else None

    best = None
    for sid, entry in catalog.items():
        if not isinstance(entry, dict):
            continue
        sn = entry.get("standard_names", {}) or {}
        ind = (sn.get("indicator") or "").lower()
        act = (sn.get("activity") or "").lower()
        seas = (sn.get("seasonal") or sn.get("seasonality") or "").lower()
        mt = (sn.get("metric_type") or "").lower()

        if indicator_l and ind != indicator_l:
            continue
        if activity_l and act and act != activity_l:
            continue
        if seasonal_l and seas and seas != seasonal_l:
            continue
        if metric_type_l:
            if not mt or mt != metric_type_l:
                continue

        best = {
            "id": sid,
            "title": entry.get("title"),
            "source_url": entry.get("source_url"),
        }
        break

    return best

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
        
        # Cargar labels
        self.intent_label_lst = get_intent_labels(self.args)
        self.indicator_label_lst = get_indicator_labels(self.args)
        self.metric_type_label_lst = get_metric_type_labels(self.args)
        self.calc_mode_label_lst = get_calc_mode_labels(self.args)
        self.seasonal_label_lst = get_seasonal_labels(self.args)
        self.req_form_label_lst = get_req_form_labels(self.args)
        self.context_mode_label_lst = get_context_mode_labels(self.args)
        self.activity_label_lst = get_activity_labels(self.args)
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
            indicator_label_lst=self.indicator_label_lst,
            metric_type_label_lst=self.metric_type_label_lst,
            calc_mode_label_lst=self.calc_mode_label_lst,
            seasonal_label_lst=self.seasonal_label_lst,
            req_form_label_lst=self.req_form_label_lst,
            context_mode_label_lst=self.context_mode_label_lst,
            activity_label_lst=self.activity_label_lst,
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
                'indicator_label_ids': None,
                'metric_type_label_ids': None,
                'calc_mode_label_ids': None,
                'seasonal_label_ids': None,
                'req_form_label_ids': None,
                'context_mode_label_ids': None,
                'activity_label_ids': None,
                'slot_labels_ids': None
            }
            outputs = self.model(**inputs)
            logits = outputs[1]
            (
                intent_logits,
                indicator_logits,
                metric_type_logits,
                calc_mode_logits,
                seasonal_logits,
                req_form_logits,
                context_mode_logits,
                activity_logits,
                slot_logits,
            ) = logits
            
            # Intent
            intent_pred = intent_logits.argmax(dim=-1).item()
            intent_label = self.intent_label_lst[intent_pred]
            intent_confidence = torch.softmax(intent_logits, dim=-1)[0][intent_pred].item()

            def predict_head(logit_tensor, label_list):
                """Return (label, confidence) for a classification head."""
                if logit_tensor is None or logit_tensor.numel() == 0:
                    return None, None
                pred_idx = logit_tensor.argmax(dim=-1).item()
                label = label_list[pred_idx] if label_list else None
                confidence = torch.softmax(logit_tensor, dim=-1)[0][pred_idx].item()
                return label, confidence

            indicator_label, indicator_confidence = predict_head(indicator_logits, self.indicator_label_lst)
            metric_type_label, metric_type_confidence = predict_head(metric_type_logits, self.metric_type_label_lst)
            calc_mode_label, calc_mode_confidence = predict_head(calc_mode_logits, self.calc_mode_label_lst)
            seasonal_label, seasonal_confidence = predict_head(seasonal_logits, self.seasonal_label_lst)
            req_form_label, req_form_confidence = predict_head(req_form_logits, self.req_form_label_lst)
            context_mode_label, context_mode_confidence = predict_head(context_mode_logits, self.context_mode_label_lst)
            activity_label, activity_confidence = predict_head(activity_logits, self.activity_label_lst)

            # Fechas según req_form
            point_period = None
            point_period_meta = None
            range_period = None
            range_period_meta = None
            if req_form_label == 'point':
                parsed = extract_month_period(text)
                if parsed:
                    point_period = parsed.get('period')
                    point_period_meta = parsed
            elif req_form_label == 'range':
                parsed_range = extract_month_range(text)
                if parsed_range:
                    range_period = {
                        'start': parsed_range['start']['period'],
                        'end': parsed_range['end']['period'],
                    }
                    range_period_meta = parsed_range

            series_info = None
            series_note = None
        if metric_type_label == 'contribution' and (not activity_label or activity_label == 'none'):
            if calc_mode_label == 'yoy':
                series_note = "Buscar actividad mas relevante en imacec contrib. a/a"
            elif calc_mode_label == 'prev_period':
                series_note = "Buscar actividad mas relevante en imacec contrib. mes/mes"
            else:
                series_note = "Buscar actividad mas relevante en imacec contribución"
        else:
            series_info = _find_series(indicator_label, activity_label, seasonal_label, metric_type_label)
            slot_labels = []
        
        return {
            'text': text,
            'words': words,
            'intent': intent_label,
            'intent_confidence': intent_confidence,
            'indicator': indicator_label,
            'indicator_confidence': indicator_confidence,
            'metric_type': metric_type_label,
            'metric_type_confidence': metric_type_confidence,
            'calc_mode': calc_mode_label,
            'calc_mode_confidence': calc_mode_confidence,
            'seasonal': seasonal_label,
            'seasonal_confidence': seasonal_confidence,
            'req_form': req_form_label,
            'req_form_confidence': req_form_confidence,
            'context_mode': context_mode_label,
            'context_mode_confidence': context_mode_confidence,
            'activity': activity_label,
            'activity_confidence': activity_confidence,
            'point_period': point_period,
            'point_period_meta': point_period_meta,
            'range_period': range_period,
            'range_period_meta': range_period_meta,
            'series_info': series_info,
            'series_note': series_note,
            'slots': slot_labels
        }
    
    def format_result(self, result):
        """Formatea el resultado para mostrar en consola."""
        output = []
        output.append(f"\n{'='*60}")
        output.append(f"Texto: {result['text']}")
        output.append(f"{'='*60}")
        output.append(f"\nIntención: {result['intent']} (confianza: {result['intent_confidence']:.2%})")
        output.append(f"Indicador: {result.get('indicator') or 'N/A'}" + (f" (confianza: {result.get('indicator_confidence', 0):.2%})" if result.get('indicator_confidence') is not None else ""))
        output.append(f"Tipo de métrica: {result.get('metric_type') or 'N/A'}" + (f" (confianza: {result.get('metric_type_confidence', 0):.2%})" if result.get('metric_type_confidence') is not None else ""))
        output.append(f"Modo de cálculo: {result.get('calc_mode') or 'N/A'}" + (f" (confianza: {result.get('calc_mode_confidence', 0):.2%})" if result.get('calc_mode_confidence') is not None else ""))
        output.append(f"Temporalidad: {result.get('seasonal') or 'N/A'}" + (f" (confianza: {result.get('seasonal_confidence', 0):.2%})" if result.get('seasonal_confidence') is not None else ""))
        output.append(f"Formato de req.: {result.get('req_form') or 'N/A'}" + (f" (confianza: {result.get('req_form_confidence', 0):.2%})" if result.get('req_form_confidence') is not None else ""))
        output.append(f"Contexto: {result.get('context_mode') or 'N/A'}" + (f" (confianza: {result.get('context_mode_confidence', 0):.2%})" if result.get('context_mode_confidence') is not None else ""))
        output.append(f"Actividad: {result.get('activity') or 'N/A'}" + (f" (confianza: {result.get('activity_confidence', 0):.2%})" if result.get('activity_confidence') is not None else ""))

        # Detalle de serie
        series_info = result.get('series_info')
        series_note = result.get('series_note')
        if series_info:
            output.append("\n" + "═"*60)
            output.append(f"Serie: {series_info.get('title') or 'N/A'}")
            if series_info.get('id'):
                output.append(f"  · Código: {series_info['id']}")
            if result.get('calc_mode') == 'yoy':
                output.append("═"*60)
                output.append("Variación respecto al año anterior")
            elif result.get('calc_mode') == 'prev_period':
                output.append("═"*60)
                output.append("Variación respecto al periodo anterior")
        if series_note:
            output.append("\n" + "═"*60)
            output.append(series_note)
        
        output.append("\n" + "═"*60)
        req_form = result.get('req_form')
        if req_form == 'latest':
            output.append("Fecha: última cifra disponible")
        elif req_form == 'point':
            date_str = result.get('point_period') or 'N/A'
            output.append(f"Fecha: {date_str}")
        elif req_form == 'range':
            rng = result.get('range_period')
            range_str = f"{rng['start']} a {rng['end']}" if rng else "N/A"
            output.append(f"Fecha: {range_str}")
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
