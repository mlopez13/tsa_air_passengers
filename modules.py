from torch import nn


class LSTMNet(nn.Module):
    def __init__(
        self, dim_input, dim_recurrent, num_layers, dim_output, is_dropout=False
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=dim_input,
            hidden_size=dim_recurrent,
            num_layers=num_layers,
            batch_first=True,
        )
        self.linear = nn.Linear(dim_recurrent, dim_output)
        self.is_dropout = is_dropout
        self.dropout = nn.Dropout()

    def forward(self, x):
        x, _ = self.lstm(x)
        if self.is_dropout:
            x = self.dropout(x)
        y_pred = self.linear(x)
        return y_pred
