import torch.nn as nn


class Classifier(nn.Module):
    """Dropout + capa lineal. Se usa para todas las cabezas y para los slots."""

    def __init__(self, input_dim: int, num_labels: int, dropout_rate: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_labels)

    def forward(self, x):
        return self.linear(self.dropout(x))
