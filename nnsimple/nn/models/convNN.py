from torch import nn


class CNN(nn.Module):
    def __init__(self, input_size=3, hidden_size=32, out_size=2, kernel_size=3, dropout=0.0, **kwargs):
        super(CNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.kernel_size = kernel_size
        self.dropout = dropout

        self.layer1 = nn.Sequential(
            nn.Conv2d(self.input_size, self.hidden_size, self.kernel_size),
            nn.ReLU(),
            nn.Conv2d(self.hidden_size, self.hidden_size, self.kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(self.dropout)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(self.hidden_size, 2*self.hidden_size, self.kernel_size),
            nn.ReLU(),
            nn.Conv2d(2*self.hidden_size, 2*self.hidden_size, self.kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(self.dropout)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(2*self.hidden_size * 5 * 5, 512),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(512, self.out_size)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(-1, 2*self.hidden_size * 5 * 5)
        out = self.fc3(out)
        return out
