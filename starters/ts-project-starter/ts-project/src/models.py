import torch.nn as nn
import torch

def _act(name: str):
    name = name.lower()
    if name == "relu": return nn.ReLU()
    if name == "tanh": return nn.Tanh()
    return nn.ReLU()

class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dims=(256,128), dropout=0.1, activation="relu"):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), _act(activation), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)  # B

class LSTMRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim=128, num_layers=1, bidirectional=False, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional, dropout=0.0 if num_layers==1 else dropout)
        mult = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim*mult, 1)
    def forward(self, x):         # x: B x T x F
        out, (h_n, c_n) = self.lstm(x)
        if self.lstm.bidirectional:
            h = torch.cat((h_n[-2], h_n[-1]), dim=1)  # B x 2H
        else:
            h = h_n[-1]                                # B x H
        h = self.dropout(h)
        y = self.fc(h).squeeze(-1)                     # B
        return y
