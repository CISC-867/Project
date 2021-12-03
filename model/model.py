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

        self.lstm1 = nn.LSTM(
            input_size=512,
            hidden_size=32,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        self.lstm2 = nn.LSTM(
            input_size=64,
            hidden_size=512,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        self.conv4 = nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=5)
        self.batch4 = nn.BatchNorm1d(num_features=512)
        self.conv5 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5)
        self.batch5 = nn.BatchNorm1d(num_features=512)
        self.conv6 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5)
        self.batch6 = nn.BatchNorm1d(num_features=512)

        self.lstm3 = nn.LSTM(
            input_size=512,
            hidden_size=1024,
            num_layers=3,
            bidirectional=True,
            batch_first=True
        )

        self.dense = nn.Linear(
            in_features=2048,
            out_features=80
        )

    def forward(self, mel_spectro):
        x = mel_spectro
        print(x.shape, "input")

        x = self.batch1(F.relu(self.conv1(x)))
        print(x.shape, "conv-batch-1 out")

        x = self.batch2(F.relu(self.conv2(x)))
        print(x.shape, "conv-batch-2 out")

        x = self.batch3(F.relu(self.conv3(x)))
        print(x.shape, "conv-batch-3 out")

        x = x.transpose(2,1)
        print(x.shape, "transpose out")

        x, _ = self.lstm1(x) # ignore final hidden state
        print(x.shape, "lstm-1 out")

        # downsampling, take every 32nd sample
        reduction_factor = 32 
        x = x[:,::reduction_factor,:]
        print(x.shape, "downsampled out")

        # upsampling, repeat every sample 32 times
        x = x.repeat(1,1,reduction_factor).reshape(x.size(0),-1,x.size(2))
        print(x.shape, "upsampled out")

        x, _ = self.lstm2(x)
        print(x.shape, "lstm-2 out")

        x = x.transpose(2,1)
        print(x.shape, "transpose out")

        x = self.batch1(F.relu(self.conv4(x)))
        print(x.shape, "conv-batch-4 out")

        x = self.batch2(F.relu(self.conv5(x)))
        print(x.shape, "conv-batch-5 out")

        x = self.batch3(F.relu(self.conv6(x)))
        print(x.shape, "conv-batch-6 out")

        x = x.transpose(2,1)
        print(x.shape, "transpose out")

        x, _ = self.lstm3(x)
        print(x.shape, "lstm-3 out")

        x = self.dense(x)
        print(x.shape, "dense out")

        return x