import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoModel
from .module import CalcModeClassifier, ActivityClassifier, RegionClassifier, InvestmentClassifier, ReqFormClassifier, SlotClassifier

try:
    from torchcrf import CRF
except ImportError:
    CRF = None

class JointBERT(PreTrainedModel):
    def __init__(self, config, args, calc_mode_label_lst, activity_label_lst, region_label_lst, investment_label_lst, req_form_label_lst, slot_label_lst):
        super(JointBERT, self).__init__(config)
        self.args = args
        
        self.num_calc_mode_labels = len(calc_mode_label_lst)
        self.num_activity_labels = len(activity_label_lst)
        self.num_region_labels = len(region_label_lst)
        self.num_investment_labels = len(investment_label_lst)
        self.num_req_form_labels = len(req_form_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        
        # Usar AutoModel para soportar cualquier encoder transformer
        self.encoder = AutoModel.from_pretrained(args.model_name_or_path, config=config)

        self.calc_mode_classifier = CalcModeClassifier(config.hidden_size, self.num_calc_mode_labels, args.dropout_rate)
        self.activity_classifier = ActivityClassifier(config.hidden_size, self.num_activity_labels, args.dropout_rate)
        self.region_classifier = RegionClassifier(config.hidden_size, self.num_region_labels, args.dropout_rate)
        self.investment_classifier = InvestmentClassifier(config.hidden_size, self.num_investment_labels, args.dropout_rate)
        self.req_form_classifier = ReqFormClassifier(config.hidden_size, self.num_req_form_labels, args.dropout_rate)
        self.slot_classifier = SlotClassifier(config.hidden_size, self.num_slot_labels, args.dropout_rate)

        if args.use_crf:
            if CRF is None:
                raise ImportError("torchcrf no está instalado. Instala con: pip install pytorch-crf o ejecuta sin --use_crf")
            crf_init_errors = []
            for init_fn in (
                lambda: CRF(self.num_slot_labels, pad_idx=None, use_gpu=False),
                lambda: CRF(self.num_slot_labels, batch_first=True),
                lambda: CRF(num_tags=self.num_slot_labels, batch_first=True),
                lambda: CRF(self.num_slot_labels),
                lambda: CRF(num_tags=self.num_slot_labels),
            ):
                try:
                    self.crf = init_fn()
                    break
                except TypeError as e:
                    crf_init_errors.append(str(e))
            else:
                raise TypeError("No se pudo inicializar CRF con las firmas conocidas: " + " | ".join(crf_init_errors))

    def forward(self, input_ids, attention_mask, token_type_ids=None,
        calc_mode_label_ids=None, activity_label_ids=None, region_label_ids=None, investment_label_ids=None, req_form_label_ids=None, slot_labels_ids=None):
        outputs = self.encoder(input_ids, attention_mask=attention_mask,
                              token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = getattr(outputs, "pooler_output", None)
        if pooled_output is None:
            if len(outputs) > 1 and outputs[1] is not None and getattr(outputs[1], "dim", lambda: 0)() == 2:
                pooled_output = outputs[1]
            else:
                pooled_output = sequence_output[:, 0]

        calc_mode_logits = self.calc_mode_classifier(pooled_output)
        activity_logits = self.activity_classifier(pooled_output)
        region_logits = self.region_classifier(pooled_output)
        investment_logits = self.investment_classifier(pooled_output)
        req_form_logits = self.req_form_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        total_loss = 0

        def _get_weight(head_name):
            """Retorna class weights registrados como buffer, o None."""
            buf_name = f"{head_name}_class_weights"
            w = getattr(self, buf_name, None)
            return w

        # 1. Calc Mode CrossEntropy
        if calc_mode_label_ids is not None:
            calc_mode_loss_fct = nn.CrossEntropyLoss(weight=_get_weight('calc_mode'))
            calc_mode_loss = calc_mode_loss_fct(calc_mode_logits.view(-1, self.num_calc_mode_labels), calc_mode_label_ids.view(-1))
            total_loss += calc_mode_loss

        # 2. Activity CrossEntropy
        if activity_label_ids is not None:
            activity_loss_fct = nn.CrossEntropyLoss(weight=_get_weight('activity'))
            activity_loss = activity_loss_fct(activity_logits.view(-1, self.num_activity_labels), activity_label_ids.view(-1))
            total_loss += activity_loss
            
        # 3. Region CrossEntropy
        if region_label_ids is not None:
            region_loss_fct = nn.CrossEntropyLoss(weight=_get_weight('region'))
            region_loss = region_loss_fct(region_logits.view(-1, self.num_region_labels), region_label_ids.view(-1))
            total_loss += region_loss

        # 4. Investment CrossEntropy
        if investment_label_ids is not None:
            investment_loss_fct = nn.CrossEntropyLoss(weight=_get_weight('investment'))
            investment_loss = investment_loss_fct(investment_logits.view(-1, self.num_investment_labels), investment_label_ids.view(-1))
            total_loss += investment_loss

        # 5. Req Form CrossEntropy
        if req_form_label_ids is not None:
            req_form_loss_fct = nn.CrossEntropyLoss(weight=_get_weight('req_form'))
            req_form_loss = req_form_loss_fct(req_form_logits.view(-1, self.num_req_form_labels), req_form_label_ids.view(-1))
            total_loss += req_form_loss

        # 6. Slot Softmax
        if slot_labels_ids is not None and self.args.slot_loss_coef != 0:
            if self.args.use_crf:
                # CRF doesn't handle ignore_index (-100), so we replace it with PAD (0)
                slot_labels_ids_crf = slot_labels_ids.clone()
                slot_labels_ids_crf[slot_labels_ids_crf == self.args.ignore_index] = 0
                if hasattr(self.crf, 'viterbi_decode'):
                    # TorchCRF API: forward returns log-likelihood per batch item
                    slot_loss = -self.crf(slot_logits, slot_labels_ids_crf, attention_mask.bool()).mean()
                else:
                    # pytorch-crf API
                    slot_loss = self.crf(slot_logits, slot_labels_ids_crf, mask=attention_mask.bool(), reduction='mean')
                    slot_loss = -1 * slot_loss  # negative log-likelihood
            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                    active_labels = slot_labels_ids.view(-1)[active_loss]
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))
            total_loss += self.args.slot_loss_coef * slot_loss

        outputs = ((calc_mode_logits, activity_logits, region_logits, investment_logits, req_form_logits, slot_logits),) + outputs[2:]  # add hidden states and attention if they are here 

        outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of all classifier logits