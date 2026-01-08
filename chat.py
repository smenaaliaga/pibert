"""
Chat interactivo con PIBot Joint BERT.

Uso:
    python chat.py --model_dir model_out/pibot_model_beto_v2
"""

import argparse
import os
import logging
import torch
import json
from datetime import date
from pathlib import Path
from transformers import BertTokenizer
from colorama import init, Fore, Style

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
from util.normalizer_date import extract_month_period, extract_month_range

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

# Inicializar colorama para Windows
init(autoreset=True)

logger = logging.getLogger(__name__)


class ChatBot:
    def __init__(self, model_dir, no_cuda=False):
        self.model_dir = model_dir
        # Detectar dispositivo: CUDA > MPS > CPU
        if torch.cuda.is_available() and not no_cuda:
            self.device = "cuda"
        elif torch.backends.mps.is_available() and not no_cuda:
            self.device = "mps"
        else:
            self.device = "cpu"
        
        print(f"\n{Fore.CYAN}Cargando modelo PIBot...{Style.RESET_ALL}")
        
        # Cargar args del entrenamiento
        self.args = torch.load(os.path.join(model_dir, 'training_args.bin'), weights_only=False)
        
        # Cargar labels (todos los heads)
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
            slot_label_lst=self.slot_label_lst,
        )
        self.model.to(self.device)
        self.model.eval()
        
        print(f"{Fore.GREEN}✓ Modelo cargado correctamente{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}  Dispositivo: {self.device.upper()}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}  Intenciones: {len(self.intent_label_lst)}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}  Indicadores: {len(self.indicator_label_lst)}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}  Métrica: {len(self.metric_type_label_lst)}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}  Calc mode: {len(self.calc_mode_label_lst)}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}  Temporalidad: {len(self.seasonal_label_lst)}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}  Req form: {len(self.req_form_label_lst)}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}  Contexto: {len(self.context_mode_label_lst)}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}  Actividad: {len(self.activity_label_lst)}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}  Slots: {len(self.slot_label_lst)}{Style.RESET_ALL}\n")
    
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
                'slot_labels_ids': None,
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

            # Slots deshabilitados
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
            'slots': slot_labels,
        }
    
    def format_response(self, result):
        """Formatea la respuesta de manera amigable (slots deshabilitados)."""
        print(f"\n{Fore.CYAN}╔{'═'*70}╗")
        print(f"{Fore.CYAN}║ {Fore.WHITE}RESPUESTA{' '*60}{Fore.CYAN}║")
        print(f"{Fore.CYAN}╠{'═'*70}╣")

        def line(label, value, conf):
            text = f"{label}: {value or 'N/A'}"
            conf_text = f" ({conf:.1%})" if conf is not None else ""
            full = text + conf_text
            padding = 68 - len(full)
            padding = padding if padding > 0 else 1
            print(f"{Fore.CYAN}║ {Fore.YELLOW}{full}{' '*padding}{Fore.CYAN}║")

        line("Intención", result['intent'], result['intent_confidence'])
        line("Indicador", result.get('indicator'), result.get('indicator_confidence'))
        line("Tipo métrica", result.get('metric_type'), result.get('metric_type_confidence'))
        line("Calc mode", result.get('calc_mode'), result.get('calc_mode_confidence'))
        line("Temporalidad", result.get('seasonal'), result.get('seasonal_confidence'))
        line("Req form", result.get('req_form'), result.get('req_form_confidence'))
        line("Contexto", result.get('context_mode'), result.get('context_mode_confidence'))
        line("Actividad", result.get('activity'), result.get('activity_confidence'))

        series_info = result.get('series_info')
        series_note = result.get('series_note')
        if series_info:
            print(f"{Fore.CYAN}╠{'═'*70}╣")
            title = series_info.get('title') or 'N/A'
            sid = series_info.get('id') or 'N/A'
            padding_t = 68 - len(title) - len("Serie: ")
            padding_t = padding_t if padding_t > 0 else 1
            print(f"{Fore.CYAN}║ {Fore.GREEN}Serie: {title}{' '*padding_t}{Fore.CYAN}║")
            padding_id = 68 - len(sid) - len("Código: ")
            padding_id = padding_id if padding_id > 0 else 1
            print(f"{Fore.CYAN}║ {Fore.GREEN}Código: {sid}{' '*padding_id}{Fore.CYAN}║")
            if result.get('calc_mode') == 'yoy':
                print(f"{Fore.CYAN}╠{'═'*70}╣")
                text = "Variación respecto al año anterior"
                padding = 68 - len(text)
                padding = padding if padding > 0 else 1
                print(f"{Fore.CYAN}║ {Fore.GREEN}{text}{' '*padding}{Fore.CYAN}║")
            elif result.get('calc_mode') == 'prev_period':
                print(f"{Fore.CYAN}╠{'═'*70}╣")
                text = "Variación respecto al periodo anterior"
                padding = 68 - len(text)
                padding = padding if padding > 0 else 1
                print(f"{Fore.CYAN}║ {Fore.GREEN}{text}{' '*padding}{Fore.CYAN}║")
        if series_note:
            print(f"{Fore.CYAN}╠{'═'*70}╣")
            padding = 68 - len(series_note)
            padding = padding if padding > 0 else 1
            print(f"{Fore.CYAN}║ {Fore.GREEN}{series_note}{' '*padding}{Fore.CYAN}║")
        
        # Fecha al final
        print(f"{Fore.CYAN}╠{'═'*70}╣")
        req_form = result.get('req_form')
        if req_form == 'latest':
            text = "última cifra disponible"
            label = "Fecha: "
        elif req_form == 'point':
            text = result.get('point_period') or 'N/A'
            label = "Fecha: "
        elif req_form == 'range':
            rng = result.get('range_period')
            text = f"{rng['start']} a {rng['end']}" if rng else "N/A"
            label = "Fecha: "
        else:
            text = None
            label = None
        
        if text is not None:
            padding = 68 - len(text) - len(label)
            padding = padding if padding > 0 else 1
            print(f"{Fore.CYAN}║ {Fore.GREEN}{label}{text}{' '*padding}{Fore.CYAN}║")
        
        print(f"{Fore.CYAN}╚{'═'*70}╝{Style.RESET_ALL}\n")
    
    def run(self):
        """Inicia el chat interactivo."""
        print(f"{Fore.CYAN}{'='*60}")
        print(f"{Fore.WHITE}     PIBot - Chat Interactivo con Joint BERT")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"\n{Fore.YELLOW}Escribe tus consultas sobre indicadores económicos.")
        print(f"Comandos especiales:")
        print(f"  • {Fore.GREEN}'salir'{Style.RESET_ALL} / {Fore.GREEN}'exit'{Style.RESET_ALL} - Terminar el chat")
        print(f"  • {Fore.GREEN}'ayuda'{Style.RESET_ALL} / {Fore.GREEN}'help'{Style.RESET_ALL} - Mostrar ayuda")
        print(f"  • {Fore.GREEN}'ejemplos'{Style.RESET_ALL} - Ver ejemplos de consultas{Style.RESET_ALL}\n")
        
        while True:
            try:
                # Prompt
                user_input = input(f"{Fore.BLUE}Tu consulta >{Style.RESET_ALL} ").strip()
                
                if not user_input:
                    continue
                
                # Comandos especiales
                if user_input.lower() in ['salir', 'exit', 'quit', 'q']:
                    print(f"\n{Fore.GREEN}¡Hasta luego!{Style.RESET_ALL}\n")
                    break
                
                if user_input.lower() in ['ayuda', 'help', '?']:
                    self._show_help()
                    continue
                
                if user_input.lower() in ['ejemplos', 'examples']:
                    self._show_examples()
                    continue
                
                # Procesar consulta
                result = self.predict(user_input)
                self.format_response(result)
                
            except KeyboardInterrupt:
                print(f"\n\n{Fore.GREEN}¡Hasta luego!{Style.RESET_ALL}\n")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"\n{Fore.RED}Error al procesar la consulta: {e}{Style.RESET_ALL}\n")
    
    def _show_help(self):
        """Muestra ayuda."""
        print(f"\n{Fore.CYAN}╔{'═'*70}╗")
        print(f"{Fore.CYAN}║ {Fore.WHITE}AYUDA{' '*64}{Fore.CYAN}║")
        print(f"{Fore.CYAN}╠{'═'*70}╣")
        print(f"{Fore.CYAN}║ {Fore.YELLOW}Este chatbot puede ayudarte con consultas sobre:{' '*22}{Fore.CYAN}║")
        print(f"{Fore.CYAN}║   {Fore.GREEN}• Indicadores económicos (PIB, IMACEC, etc.){' '*23}{Fore.CYAN}║")
        print(f"{Fore.CYAN}║   {Fore.GREEN}• Fechas y períodos específicos{' '*37}{Fore.CYAN}║")
        print(f"{Fore.CYAN}║   {Fore.GREEN}• Variaciones y comparaciones{' '*39}{Fore.CYAN}║")
        print(f"{Fore.CYAN}║{' '*70}║")
        print(f"{Fore.CYAN}║ {Fore.YELLOW}El modelo detecta:{' '*51}{Fore.CYAN}║")
        print(f"{Fore.CYAN}║   {Fore.MAGENTA}• Intención de tu consulta{' '*42}{Fore.CYAN}║")
        print(f"{Fore.CYAN}║   {Fore.MAGENTA}• Entidades relevantes (fechas, indicadores){' '*24}{Fore.CYAN}║")
        print(f"{Fore.CYAN}╚{'═'*70}╝{Style.RESET_ALL}\n")
    
    def _show_examples(self):
        """Muestra ejemplos de consultas."""
        print(f"\n{Fore.CYAN}╔{'═'*70}╗")
        print(f"{Fore.CYAN}║ {Fore.WHITE}EJEMPLOS DE CONSULTAS{' '*48}{Fore.CYAN}║")
        print(f"{Fore.CYAN}╠{'═'*70}╣")
        print(f"{Fore.CYAN}║ {Fore.GREEN}• cual fue el imacec de enero 2023{' '*33}{Fore.CYAN}║")
        print(f"{Fore.CYAN}║ {Fore.GREEN}• dame el ultimo pib{' '*48}{Fore.CYAN}║")
        print(f"{Fore.CYAN}║ {Fore.GREEN}• como estuvo el imacec el mes pasado{' '*31}{Fore.CYAN}║")
        print(f"{Fore.CYAN}║ {Fore.GREEN}• pib del segundo trimestre{' '*41}{Fore.CYAN}║")
        print(f"{Fore.CYAN}║ {Fore.GREEN}• variacion del imacec de diciembre{' '*32}{Fore.CYAN}║")
        print(f"{Fore.CYAN}╚{'═'*70}╝{Style.RESET_ALL}\n")


def main():
    init_logger()
    
    parser = argparse.ArgumentParser(description='Chat interactivo con PIBot Joint BERT')
    parser.add_argument('--model_dir', type=str, required=True, help='Directorio del modelo entrenado')
    parser.add_argument('--no_cuda', action='store_true', help='No usar CUDA')
    
    args = parser.parse_args()
    
    # Cargar chatbot
    try:
        chatbot = ChatBot(args.model_dir, args.no_cuda)
        chatbot.run()
    except Exception as e:
        logger.error(f"Error cargando modelo: {e}")
        print(f"\n{Fore.RED}Error: No se pudo cargar el modelo desde {args.model_dir}")
        print(f"Detalle: {e}{Style.RESET_ALL}\n")
        return


if __name__ == '__main__':
    main()
