import torch.nn as nn

class CalcModeClassifier(nn.Module):
    def __init__(self, input_dim, num_calc_mode_labels, dropout_rate=0.):
        super(CalcModeClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_calc_mode_labels)

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
    
class RegionClassifier(nn.Module):
    def __init__(self, input_dim, num_region_labels, dropout_rate=0.):
        super(RegionClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_region_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)

class InvestmentClassifier(nn.Module):
    def __init__(self, input_dim, num_investment_labels, dropout_rate=0.):
        super(InvestmentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_investment_labels)

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

class SlotClassifier(nn.Module):
    def __init__(self, input_dim, num_slot_labels, dropout_rate=0.):
        super(SlotClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_slot_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)
    