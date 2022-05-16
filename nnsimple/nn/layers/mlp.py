import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 out_size,
                 n_layers=1,
                 activation=nn.ReLU,
                 dropout=0.0,
                 lazy=False):
        """
        Multi Layer Perceptron. It is composed by n_layers of (Linear, activation, Dropout)


        :param input_size: The number of expected features in the input x
        :param hidden_size: The number of features in the hidden state h
        :param out_size: the number of classes in the output
        :param n_layers: number of layers in the MLP, defaults to 1 (optional)
        :param activation: The activation function to use
        :param dropout: The dropout rate to use, defaults 0.0
        :param lazy: If True, the network will use the nn.LazyLayer() which will infer the input dimension.
                    if True the parameter input_size will not be used.
        """
        super(MLP, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.out_size = out_size
        self.activation = activation
        self.dropout = dropout
        self.lazy = lazy

        lin = []
        up_scale = nn.LazyLinear(self.hidden_size) if self.lazy else nn.Linear(self.input_size, self.hidden_size)
        for i in range(n_layers):
            linear = up_scale if i == 0 else nn.Linear(self.hidden_size, self.hidden_size)
            lin.append(linear)
            lin.append(activation)
            lin.append(nn.Dropout(self.dropout))

        lin.append(nn.Linear(self.hidden_size, self.out_size))

        self.out = nn.Sequential(*lin)

    def forward(self, x):
        return self.out(x)

