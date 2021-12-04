import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import tqdm.notebook as tqdm

class Trainer:
    def __init__(self, model, vocoder, device) -> None:
        self.model = model
        self.vocoder = vocoder
        self.device = device
        self.model_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  
        self.vocoder_optimizer = torch.optim.Adam(vocoder.parameters(), lr=0.001)  
        self.spectro_loss_func = nn.L1Loss()
        self.vocoder_loss_func = F.cross_entropy

    def train_step(self, x, y):
        y_pred_spectro = self.model(x)
        y_pred_spectro = y_pred_spectro.transpose(2,1)
        spectro_loss = self.spectro_loss_func(y_pred_spectro, x)
        
        # for some reason our dataset returns 129 instead of 128 time slices
        # lets just chop the last one off for now.
        x = x[:,:,:128]
        
        y_pred_wav = self.vocoder(y_pred_spectro, None)
        vocoder_loss = self.vocoder_loss_func(y_pred_wav)

        total_loss = spectro_loss + vocoder_loss

        print(total_loss, spectro_loss, vocoder_loss)

        self.model_optimizer.zero_grad()
        self.vocoder_optimizer.zero_grad()
        total_loss.backward()
        self.model_optimizer.step()
        self.vocoder_optimizer.step()

        ## free tensor from gpu memory?
        ## not sure if this works
        # del y_pred_spectro

        return total_loss


    def train(self, model, dataset, device):
        text, clips, spectros = dataset[0]
        spectros = spectros[0,:,:]
        spectros = spectros.to(device)
        print(spectros.shape)
        spectros = spectros.repeat(100, 1, 1)
        print(spectros.shape)
        for i, _ in tqdm(list(enumerate(range(0,100)))):
            self.train_step(spectros, spectros)