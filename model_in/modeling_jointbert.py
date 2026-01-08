import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, BertConfig
from torchcrf import CRF
from .module import IntentClassifier, IndicatorClassifier, MetricTypeClassifier, CalcModeClassifier, SeasonalClassifier, ReqFormClassifier, ContextModeClassifier, ActivityClassifier, SlotClassifier


class JointBERT(BertPreTrainedModel):
    def __init__(self, config, args, intent_label_lst, indicator_label_lst, metric_type_label_lst, calc_mode_label_lst, 
                 seasonal_label_lst, req_form_label_lst, context_mode_label_lst, activity_label_lst, slot_label_lst):
        super(JointBERT, self).__init__(config)
        self.args = args
        
        self.num_intent_labels = len(intent_label_lst)
        self.num_indicator_labels = len(indicator_label_lst)
        self.num_metric_type_labels = len(metric_type_label_lst)
        self.num_calc_mode_labels = len(calc_mode_label_lst)
        self.num_seasonal_labels = len(seasonal_label_lst)
        self.num_req_form_labels = len(req_form_label_lst)
        self.num_context_mode_labels = len(context_mode_label_lst)
        self.num_activity_labels = len(activity_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        
        self.bert = BertModel(config=config)  # Load pretrained bert

        self.intent_classifier = IntentClassifier(config.hidden_size, self.num_intent_labels, args.dropout_rate)
        self.indicator_classifier = IndicatorClassifier(config.hidden_size, self.num_indicator_labels, args.dropout_rate)
        self.metric_type_classifier = MetricTypeClassifier(config.hidden_size, self.num_metric_type_labels, args.dropout_rate)
        self.calc_mode_classifier = CalcModeClassifier(config.hidden_size, self.num_calc_mode_labels, args.dropout_rate)
        self.seasonal_classifier = SeasonalClassifier(config.hidden_size, self.num_seasonal_labels, args.dropout_rate)
        self.req_form_classifier = ReqFormClassifier(config.hidden_size, self.num_req_form_labels, args.dropout_rate)
        self.context_mode_classifier = ContextModeClassifier(config.hidden_size, self.num_context_mode_labels, args.dropout_rate)
        self.activity_classifier = ActivityClassifier(config.hidden_size, self.num_activity_labels, args.dropout_rate)
        self.slot_classifier = SlotClassifier(config.hidden_size, self.num_slot_labels, args.dropout_rate)

        if args.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids, indicator_label_ids, metric_type_label_ids, 
                calc_mode_label_ids, seasonal_label_ids, req_form_label_ids, context_mode_label_ids, activity_label_ids, slot_labels_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        intent_logits = self.intent_classifier(pooled_output)
        indicator_logits = self.indicator_classifier(pooled_output)
        metric_type_logits = self.metric_type_classifier(pooled_output)
        calc_mode_logits = self.calc_mode_classifier(pooled_output)
        seasonal_logits = self.seasonal_classifier(pooled_output)
        req_form_logits = self.req_form_classifier(pooled_output)
        context_mode_logits = self.context_mode_classifier(pooled_output)
        activity_logits = self.activity_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        total_loss = 0
        # 1. Intent CrossEntropy
        if intent_label_ids is not None:
            intent_loss_fct = nn.CrossEntropyLoss()
            intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))
            total_loss += intent_loss
            
        # 2. Indicator CrossEntropy
        if indicator_label_ids is not None:
            indicator_loss_fct = nn.CrossEntropyLoss()
            indicator_loss = indicator_loss_fct(indicator_logits.view(-1, self.num_indicator_labels), indicator_label_ids.view(-1))
            total_loss += indicator_loss
            
        # 3. Metric Type CrossEntropy
        if metric_type_label_ids is not None:
            metric_type_loss_fct = nn.CrossEntropyLoss()
            metric_type_loss = metric_type_loss_fct(metric_type_logits.view(-1, self.num_metric_type_labels), metric_type_label_ids.view(-1))
            total_loss += metric_type_loss
            
        # 4. Calc Mode CrossEntropy
        if calc_mode_label_ids is not None:
            calc_mode_loss_fct = nn.CrossEntropyLoss()
            calc_mode_loss = calc_mode_loss_fct(calc_mode_logits.view(-1, self.num_calc_mode_labels), calc_mode_label_ids.view(-1))
            total_loss += calc_mode_loss
            
        # 5. Seasonal CrossEntropy
        if seasonal_label_ids is not None:
            seasonal_loss_fct = nn.CrossEntropyLoss()
            seasonal_loss = seasonal_loss_fct(seasonal_logits.view(-1, self.num_seasonal_labels), seasonal_label_ids.view(-1))
            total_loss += seasonal_loss
            
        # 6. Req Form CrossEntropy
        if req_form_label_ids is not None:
            req_form_loss_fct = nn.CrossEntropyLoss()
            req_form_loss = req_form_loss_fct(req_form_logits.view(-1, self.num_req_form_labels), req_form_label_ids.view(-1))
            total_loss += req_form_loss
            
        # 7. Context Mode CrossEntropy
        if context_mode_label_ids is not None:
            context_mode_loss_fct = nn.CrossEntropyLoss()
            context_mode_loss = context_mode_loss_fct(context_mode_logits.view(-1, self.num_context_mode_labels), context_mode_label_ids.view(-1))
            total_loss += context_mode_loss

        # 8. Activity CrossEntropy
        if activity_label_ids is not None:
            activity_loss_fct = nn.CrossEntropyLoss()
            activity_loss = activity_loss_fct(activity_logits.view(-1, self.num_activity_labels), activity_label_ids.view(-1))
            total_loss += activity_loss

        # 9. Slot Softmax
        if slot_labels_ids is not None:
            if self.args.use_crf:
                # CRF doesn't handle ignore_index (-100), so we replace it with PAD (0)
                slot_labels_ids_crf = slot_labels_ids.clone()
                slot_labels_ids_crf[slot_labels_ids_crf == self.args.ignore_index] = 0
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

        outputs = ((intent_logits, indicator_logits, metric_type_logits, calc_mode_logits, seasonal_logits, req_form_logits, context_mode_logits, activity_logits, slot_logits),) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits