import torch
import torch.nn as nn

class SignLanguageLSTM(nn.Module):
    def __init__(self, input_dim=162, hidden_dim=64, num_layers=2, num_classes=25):
        super(SignLanguageLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x is (batch_size, seq_length, input_dim)
        out, (hn, cn) = self.lstm(x)
        # Get the output from the last timestep
        out = self.fc(out[:, -1, :])
        return out
