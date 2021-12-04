import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import tqdm.notebook as tqdm

def get_vocoder_model():
    import sys
    import os.path

    sys.path.append(os.path.abspath("ForwardTacotron"))
    import models.fatchord_version

    return models.fatchord_version.WaveRNN(
        rnn_dims=512,
        fc_dims=512,
        bits=9, # OrigAuthor: bit depth of signal
        pad=2, # OrigAuthor: this will pad the input so that the resnet can 'see' wider than input length
        upsample_factors=(5, 5, 8), # OrigAuthor: NB - this needs to correctly factorise hop_length
        feat_dims=80,
        compute_dims=128,
        res_out_dims=128,
        res_blocks=10,
        hop_length=200,
        sample_rate=16000,
        mode="RAW", # OrigAuthor: either 'RAW' (softmax on raw bits) or 'MOL' (sample from mixture of logistics)
    )

def get_spectrogram_model():
    import model.model
    return model.model.SpectrogramModel()

def get_dataset():
    import data.VCTK
    return data.VCTK.VCTKDataset("VCTK-Corpus-smaller/")

class Trainer:
    def __init__(self,
        model=None,
        vocoder=None,
        device=None,
        checkpoint=None,
        load_from_checkpoint=False
    ) -> None:
        if device is None:
            device = torch.device("cpu")
            print(f"Using default device: {device}")

        if model is None:
            model = get_spectrogram_model()
            print("Using default model")

        if vocoder is None:
            vocoder = get_vocoder_model()
            print("Using default vocoder")

        if load_from_checkpoint:
            if checkpoint is None:
                raise ValueError("Checkpoint can't be None when loading")
            print(f"Loading models from checkpoint {checkpoint}")
            Trainer.load_from(model, f"model{checkpoint}.pth")
            Trainer.load_from(vocoder, f"model{checkpoint}_vocoder.pth")
            print("Models loaded")

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


    def train(
        self,
        dataset=None,
        epochs=10,
        save_every_n=10,
    ):
        if dataset is None:
            dataset = get_dataset()

        epochs = 10
        save_every_n=10

        print(f"Beginning training {epochs} epochs, saving every {save_every_n}.")
        
        for epoch in range(epochs):
            for i, entry in enumerate(dataset):
                text, clips, spectros = entry
                total, spectro, vocoder = self.train_step(clips, spectros)
                print("Total loss: ", total)
                if i+1 % save_every_n == 0:
                    Trainer.save_as(spectrogram_model, f"model{self.checkpoint+i}.pth")
                    Trainer.save_as(vocoder_model, f"model{self.checkpoint+i}_vocoder.pth")
                    self.checkpoint += i

    @classmethod
    def save_as(self, model, name, dir="checkpoints"):
        import pathlib
        import os.path
        save_dir = pathlib.Path(dir)
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / name
        if os.path.exists(save_path):
            raise Exception("Won't overwrite existing models")
        else:
            torch.save(model.state_dict(), save_path)

    @classmethod
    def load_from(self, model, name, dir="checkpoints"):
        import pathlib
        import os.path
        save_dir = pathlib.Path(dir)
        save_path = save_dir / name
        if not os.path.exists(save_path):
            raise Exception(f"Can't find checkpoint file {save_path}")
        else:
            model.load_state_dict(torch.load(save_path))
