import torch
from torch import nn


class ConvPool(nn.Module):
    def __init__(self, input_size=3, hidden_size=32, kernel_size=3, kernel_pooling=2, dropout=0.0, **kwargs):
        super(ConvPool, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_pooling = kernel_pooling
        self.kernel_size = kernel_size
        self.dropout = dropout

        self.layer1 = nn.Sequential(
            nn.Conv2d(self.input_size, self.hidden_size, self.kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=self.kernel_pooling),
            nn.Dropout(self.dropout)
        )

    def forward(self, x):
        out = self.layer1(x)
        return out
