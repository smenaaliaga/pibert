import torch.nn as nn

# class IntentClassifier(nn.Module):
#     def __init__(self, input_dim, num_intent_labels, dropout_rate=0.):
#         super(IntentClassifier, self).__init__()
#         self.dropout = nn.Dropout(dropout_rate)
#         self.linear = nn.Linear(input_dim, num_intent_labels)

#     def forward(self, x):
#         x = self.dropout(x)
#         return self.linear(x)
    
class IndicatorClassifier(nn.Module):
    def __init__(self, input_dim, num_indicator_labels, dropout_rate=0.):
        super(IndicatorClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_indicator_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)

class MetricTypeClassifier(nn.Module):
    def __init__(self, input_dim, num_metric_type_labels, dropout_rate=0.):
        super(MetricTypeClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_metric_type_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)
    
class SeasonalClassifier(nn.Module):
    def __init__(self, input_dim, num_seasonal_labels, dropout_rate=0.):
        super(SeasonalClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_seasonal_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)
    
class ActivityClassifier(nn.Module):
    def __init__(self, input_dim, num_activity_labels, dropout_rate=0.):
        super(ActivityClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_activity_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)
    
class FrequencyClassifier(nn.Module):
    def __init__(self, input_dim, num_frequency_labels, dropout_rate=0.):
        super(FrequencyClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_frequency_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)
    
class CalcModeClassifier(nn.Module):
    def __init__(self, input_dim, num_calc_mode_labels, dropout_rate=0.):
        super(CalcModeClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_calc_mode_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)
    
class ReqFormClassifier(nn.Module):
    def __init__(self, input_dim, num_req_form_labels, dropout_rate=0.):
        super(ReqFormClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_req_form_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)

# class ContextModeClassifier(nn.Module):
#     def __init__(self, input_dim, num_context_mode_labels, dropout_rate=0.):
#         super(ContextModeClassifier, self).__init__()
#         self.dropout = nn.Dropout(dropout_rate)
#         self.linear = nn.Linear(input_dim, num_context_mode_labels)

#     def forward(self, x):
#         x = self.dropout(x)
#         return self.linear(x)

# class SlotClassifier(nn.Module):
#     def __init__(self, input_dim, num_slot_labels, dropout_rate=0.):
#         super(SlotClassifier, self).__init__()
#         self.dropout = nn.Dropout(dropout_rate)
#         self.linear = nn.Linear(input_dim, num_slot_labels)

#     def forward(self, x):
#         x = self.dropout(x)
#         return self.linear(x)