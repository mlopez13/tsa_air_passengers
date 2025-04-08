from torch import nn
from torch.nn import functional as F


class LSTMNet(nn.Module):
    def __init__(self, dim_input, dim_recurrent, num_layers, dim_output):
        super().__init__()
        self.lstm = nn.LSTM(input_size=dim_input,
                            hidden_size=dim_recurrent,
                            num_layers=num_layers,
                            batch_first=True)
        self.linear = nn.Linear(dim_recurrent, dim_output)

    def forward(self, x):
        x, _ = self.lstm(x)
        # Keep only the last element of the sequence
        x = x[:, -1, :]
        y_pred = self.linear(F.relu(x))
        return y_pred
