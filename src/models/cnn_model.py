import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out):
        weights = torch.softmax(self.attn(lstm_out), dim=1)
        context = torch.sum(weights * lstm_out, dim=1)
        return context


class CNN_BiLSTM_Attention(nn.Module):
    def __init__(self, input_size, hidden_size, cnn_out=32, kernel=3):
        super().__init__()

        self.conv = nn.Conv1d(input_size, cnn_out, kernel, padding=kernel // 2)
        self.bn = nn.BatchNorm1d(cnn_out)
        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(
            input_size=cnn_out,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )

        self.attention = Attention(hidden_dim=hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)           # (batch, features, seq)
        x = self.relu(self.bn(self.conv(x)))
        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)
        context = self.attention(lstm_out)
        output = self.fc(context)
        return output
