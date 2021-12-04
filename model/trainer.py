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

    def train_step(self, audios, spectrogram):
        y_pred_spectros = self.model(spectrogram)

        # fix model returning 128 instead of 129 time steps in the spectrogram
        extra_timestep = y_pred_spectros[:,:,-1].unsqueeze(2)
        y_pred_spectros = torch.cat((y_pred_spectros, extra_timestep), dim=2)

        spectro_loss = self.spectro_loss_func(y_pred_spectros, spectrogram)
        
        # mels = y_pred_spectros.transpose(0,1)
        y_pred_spectros = torch.cat((y_pred_spectros, extra_timestep), dim=2)
        y_pred_spectros = torch.cat((y_pred_spectros, extra_timestep), dim=2)
        y_pred_spectros = torch.cat((y_pred_spectros, extra_timestep), dim=2)
        
        x = audios
        mels = y_pred_spectros
        y_pred_wavs = self.vocoder( x, mels )
        y_pred_wavs = y_pred_wavs.amax(dim=-1) # collapse 512 channel wav into 1 channel

        vocoder_loss = self.vocoder_loss_func(y_pred_wavs, audios)

        total_loss = spectro_loss + vocoder_loss


        self.model_optimizer.zero_grad()
        self.vocoder_optimizer.zero_grad()
        total_loss.backward()
        self.model_optimizer.step()
        self.vocoder_optimizer.step()

        ## free tensor from gpu memory?
        ## not sure if this works
        # del y_pred_spectro

        return total_loss, spectro_loss, vocoder_loss


    def train(self, model, dataset, device):
        text, clips, spectros = dataset[0]
        spectros = spectros[0,:,:]
        spectros = spectros.to(device)
        print(spectros.shape)
        spectros = spectros.repeat(100, 1, 1)
        print(spectros.shape)
        for i, _ in tqdm(list(enumerate(range(0,100)))):
            self.train_step(spectros, spectros)