"""
Chat interactivo con PIBot Joint BERT.

Uso:
    python chat.py --model_dir model_out/pibot_model_beto
"""

import argparse
import os
import logging
import torch
from datetime import date
from transformers import BertTokenizer
from colorama import init, Fore, Style

from utils import get_intent_labels, get_slot_labels, MODEL_CLASSES, init_logger
from util.period_normalizer import standardize_imacec_time_ref
from util.indicator_normalizer import standardize_indicator

# Inicializar colorama para Windows
init(autoreset=True)

logger = logging.getLogger(__name__)


class ChatBot:
    def __init__(self, model_dir, no_cuda=False):
        self.model_dir = model_dir
        self.device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"
        
        print(f"\n{Fore.CYAN}Cargando modelo PIBot...{Style.RESET_ALL}")
        
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
        
        print(f"{Fore.GREEN}✓ Modelo cargado correctamente{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}  Dispositivo: {self.device.upper()}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}  Intenciones: {len(self.intent_label_lst)}{Style.RESET_ALL}")
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
    
    def format_response(self, result):
        """Formatea la respuesta de manera amigable."""
        intent = result['intent']
        confidence = result['intent_confidence']
        words = result['words']
        slots = result['slots']
        
        # Extraer entidades
        entities = {}
        current_entity = None
        current_value = []
        
        for word, slot in zip(words, slots):
            if slot.startswith('B-'):
                if current_entity:
                    entities[current_entity] = ' '.join(current_value)
                current_entity = slot[2:]
                current_value = [word]
            elif slot.startswith('I-') and current_entity:
                current_value.append(word)
            else:
                if current_entity:
                    entities[current_entity] = ' '.join(current_value)
                    current_entity = None
                    current_value = []
        
        if current_entity:
            entities[current_entity] = ' '.join(current_value)
        
        # Construir respuesta
        print(f"\n{Fore.CYAN}╔{'═'*58}╗")
        print(f"{Fore.CYAN}║ {Fore.WHITE}RESPUESTA{' '*48}{Fore.CYAN}║")
        print(f"{Fore.CYAN}╠{'═'*58}╣")
        
        # Intención
        confidence_bar = self._get_confidence_bar(confidence)
        # Ancho total: 57 - "Intención: " (11) = 46 para intent + espacio final
        intent_padding = 46 - len(intent)
        print(f"{Fore.CYAN}║ {Fore.YELLOW}Intención:{Style.RESET_ALL} {Fore.GREEN}{intent}{Style.RESET_ALL}{' '*intent_padding}{Fore.CYAN}║")
        
        # Ancho total: 57 - "Confianza: " (11) - barra (20) - " " (1) = 25 para porcentaje + espacio final
        conf_text = f"{confidence:.1%}"
        conf_padding = 25 - len(conf_text)
        print(f"{Fore.CYAN}║ {Fore.YELLOW}Confianza:{Style.RESET_ALL} {confidence_bar} {conf_text}{' '*conf_padding}{Fore.CYAN}║")
        
        # Entidades
        if entities:
            print(f"{Fore.CYAN}╠{'═'*58}╣")
            print(f"{Fore.CYAN}║ {Fore.YELLOW}Entidades detectadas:{' '*36}{Fore.CYAN}║")
            for entity_type, entity_value in entities.items():
                # Ancho total: 57 - "  • " (4) - ": " (2) = 51 para entity_type + display_value + espacio final
                max_value_len = 51 - len(entity_type) - 2
                display_value = entity_value if len(entity_value) <= max_value_len else entity_value[:max_value_len-3] + "..."
                entity_padding = 51 - len(entity_type) - len(display_value)
                print(f"{Fore.CYAN}║   {Fore.MAGENTA}• {entity_type}:{Style.RESET_ALL} {display_value}{' '*entity_padding}{Fore.CYAN}║")
        
        # Normalización de período si se detectó
        if 'period' in entities:
            period_text = entities['period']
            normalized = standardize_imacec_time_ref(period_text, date.today())
            if normalized:
                print(f"{Fore.CYAN}╠{'═'*58}╣")
                print(f"{Fore.CYAN}║ {Fore.YELLOW}Período normalizado:{' '*37}{Fore.CYAN}║")
                
                # Formato de fecha según granularidad
                granularity = normalized['granularity']
                period_key = normalized['period_key']
                confidence = normalized['confidence']
                
                # Icono según confianza
                conf_icon = '✓' if confidence == 'high' else '~' if confidence == 'medium' else '?'
                
                if granularity == 'month':
                    date_str = normalized['start_date'].strftime('%B %Y')
                elif granularity == 'quarter':
                    date_str = f"Trimestre {period_key.split('-Q')[1]} de {period_key.split('-')[0]}"
                elif granularity == 'year':
                    date_str = f"Año {period_key}"
                else:
                    date_str = period_key
                
                display_str = f"{conf_icon} {date_str}"
                # Ancho total: 60 - 1 (║) - 3 (espacios) - 2 (espacios finales) - 1 (║) = 53 para display_str + padding
                padding = 53 - len(display_str)
                print(f"{Fore.CYAN}║   {Fore.CYAN}{display_str}{' '*padding}  {Fore.CYAN}║")
        
        # Normalización de indicador si se detectó
        if 'indicator' in entities:
            indicator_text = entities['indicator']
            normalized = standardize_indicator(indicator_text)
            if normalized and normalized['indicator']:
                print(f"{Fore.CYAN}╠{'═'*58}╣")
                print(f"{Fore.CYAN}║ {Fore.YELLOW}Indicador normalizado:{' '*35}{Fore.CYAN}║")
                
                indicator_name = normalized['indicator'].upper()
                
                # Nombre completo según el indicador
                if indicator_name in ['IMACEC', 'PIB']:
                    if indicator_name == 'IMACEC':
                        full_name = 'Indicador Mensual de Actividad Económica'
                    else:
                        full_name = 'Producto Interno Bruto'
                    
                    display_str = f"✓ {indicator_name} - {full_name}"
                else:
                    display_str = f"✓ {indicator_name}"
                
                # Ancho total: 60 - 1 (║) - 3 (espacios) - 2 (espacios finales) - 1 (║) = 53 para display_str + padding
                padding = 53 - len(display_str)
                print(f"{Fore.CYAN}║   {Fore.GREEN}{display_str}{' '*padding}  {Fore.CYAN}║")
        
        print(f"{Fore.CYAN}╚{'═'*58}╝{Style.RESET_ALL}\n")
    
    def _get_confidence_bar(self, confidence, width=20):
        """Genera barra de confianza visual."""
        filled = int(confidence * width)
        bar = '█' * filled + '░' * (width - filled)
        
        if confidence >= 0.8:
            color = Fore.GREEN
        elif confidence >= 0.5:
            color = Fore.YELLOW
        else:
            color = Fore.RED
        
        return f"{color}{bar}{Style.RESET_ALL}"
    
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
        print(f"\n{Fore.CYAN}╔{'═'*58}╗")
        print(f"{Fore.CYAN}║ {Fore.WHITE}AYUDA{' '*52}{Fore.CYAN}║")
        print(f"{Fore.CYAN}╠{'═'*58}╣")
        print(f"{Fore.CYAN}║ {Fore.YELLOW}Este chatbot puede ayudarte con consultas sobre:{' '*10}{Fore.CYAN}║")
        print(f"{Fore.CYAN}║   {Fore.GREEN}• Indicadores económicos (PIB, IMACEC, etc.){' '*11}{Fore.CYAN}║")
        print(f"{Fore.CYAN}║   {Fore.GREEN}• Fechas y períodos específicos{' '*25}{Fore.CYAN}║")
        print(f"{Fore.CYAN}║   {Fore.GREEN}• Variaciones y comparaciones{' '*27}{Fore.CYAN}║")
        print(f"{Fore.CYAN}║{' '*58}║")
        print(f"{Fore.CYAN}║ {Fore.YELLOW}El modelo detecta:{' '*39}{Fore.CYAN}║")
        print(f"{Fore.CYAN}║   {Fore.MAGENTA}• Intención de tu consulta{' '*30}{Fore.CYAN}║")
        print(f"{Fore.CYAN}║   {Fore.MAGENTA}• Entidades relevantes (fechas, indicadores){' '*12}{Fore.CYAN}║")
        print(f"{Fore.CYAN}╚{'═'*58}╝{Style.RESET_ALL}\n")
    
    def _show_examples(self):
        """Muestra ejemplos de consultas."""
        print(f"\n{Fore.CYAN}╔{'═'*58}╗")
        print(f"{Fore.CYAN}║ {Fore.WHITE}EJEMPLOS DE CONSULTAS{' '*36}{Fore.CYAN}║")
        print(f"{Fore.CYAN}╠{'═'*58}╣")
        print(f"{Fore.CYAN}║ {Fore.GREEN}• cual fue el imacec de enero 2023{' '*21}{Fore.CYAN}║")
        print(f"{Fore.CYAN}║ {Fore.GREEN}• dame el ultimo pib{' '*36}{Fore.CYAN}║")
        print(f"{Fore.CYAN}║ {Fore.GREEN}• como estuvo el imacec el mes pasado{' '*19}{Fore.CYAN}║")
        print(f"{Fore.CYAN}║ {Fore.GREEN}• pib del segundo trimestre{' '*29}{Fore.CYAN}║")
        print(f"{Fore.CYAN}║ {Fore.GREEN}• variacion del imacec de diciembre{' '*20}{Fore.CYAN}║")
        print(f"{Fore.CYAN}╚{'═'*58}╝{Style.RESET_ALL}\n")


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
