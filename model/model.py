import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self) -> None:
        super(MyModel, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=80, out_channels=512, kernel_size=5)
        self.batch1 = nn.BatchNorm1d(num_features=512)
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5)
        self.batch2 = nn.BatchNorm1d(num_features=512)
        self.conv3 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5)
        self.batch3 = nn.BatchNorm1d(num_features=512)

        self.lstm_hidden_size=32
        self.lstm1 = nn.LSTM(input_size=512, hidden_size=self.lstm_hidden_size, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=512, hidden_size=self.lstm_hidden_size, bidirectional=True, batch_first=True)

    def forward(self, mel_spectro):
        x = mel_spectro
        print(x.shape, end="\n\n")

        x = self.batch1(F.relu(self.conv1(x)))
        print(x.shape, end="\n\n")

        x = self.batch2(F.relu(self.conv2(x)))
        print(x.shape, end="\n\n")

        x = self.batch3(F.relu(self.conv3(x)))
        print(x.shape, end="\n\n")

        x = x.transpose(2,1)
        print(x.shape, end="\n\n")

        x, _ = self.lstm1(x)
        print(x.shape)
        x_forward = x[:, :, :self.lstm_hidden_size]
        print(x_forward.shape)
        x_backward = x[:,:, self.lstm_hidden_size:]
        print(x_backward.shape, end="\n\n")
        # x, _ = self.lstm2(x)
        return x, x_forward, x_backward