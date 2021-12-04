import torch
import torch.optim
import torch.nn as nn

import tqdm.notebook as tqdm

class Trainer:
    def __init__(self, model, device) -> None:
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  
        self.criterion = nn.L1Loss()

    def train_step(self, x, y):
        y_pred = self.model(x)
        y_pred = y_pred.transpose(2,1)
        x = x[:,:,:128]

        loss = self.criterion(y_pred, x)
        print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        del y_pred


    def train(self, model, dataset, device):
        text, clips, spectros = dataset[0]
        spectros = spectros[0,:,:]
        spectros = spectros.to(device)
        print(spectros.shape)
        spectros = spectros.repeat(100, 1, 1)
        print(spectros.shape)
        for i, _ in tqdm(list(enumerate(range(0,100)))):
            self.train_step(spectros, spectros)