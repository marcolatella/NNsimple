import torch
from torch import nn
from nnsimple.nn.layers import ConvPool, MLP


class CNN(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size_conv,
                 hidden_size_fc,
                 out_size,
                 image_shape,
                 kernel_size=3,
                 kernel_pooling = 2,
                 dropout=0.0,
                 n_layers_mlp=1,
                 stride=1,
                 padding=0):
        self.input_size = input_size
        self.hidden_size_conv = hidden_size_conv
        self.hidden_size_fc = hidden_size_fc
        self.out_size = out_size
        self.n_layers_mlp = n_layers_mlp
        self.kernel_size = kernel_size
        self.kernel_pooling = kernel_pooling
        self.image_shape = image_shape
        self.dropout = dropout
        self.stride = stride
        self.padding = padding
        super(CNN, self).__init__()

        self.conv1 = ConvPool(self.input_size,
                              self.hidden_size_conv, self.kernel_size, self.kernel_pooling, self.dropout)

        self.conv2 = ConvPool(self.hidden_size_conv,
                              2 * self.hidden_size_conv, self.kernel_size, self.kernel_pooling, self.dropout)

        self.fc = MLP(None, 512, self.out_size, activation=nn.ReLU(), lazy=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(x.size(0), -1)
        out = self.fc(out)
        return out
