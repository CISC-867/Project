import torch
import torch.nn as nn
import torch.functional as F

class MyModel(nn.Module):
    def __init__(self) -> None:
        super(MyModel, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=80, out_channels=512, kernel_size=5)
        self.batch1 = nn.BatchNorm1d(num_features=512)
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5)
        self.batch2 = nn.BatchNorm1d(num_features=512)
        self.conv3 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5)
        self.batch3 = nn.BatchNorm1d(num_features=512)

        self.lstm = nn.LSTM(input_size=512, hidden_size=32, bidirectional=True, batch_first=True)

    def forward(self, mel_spectro):
        x = mel_spectro
        x = self.batch1(F.relu(self.conv1(x)))
        x = self.batch2(F.relu(self.conv2(x)))
        x = self.batch3(F.relu(self.conv3(x)))
        return x