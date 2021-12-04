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

        if checkpoint is None:
            checkpoint = 0
        self.model = model.to(device)
        self.vocoder = vocoder.to(device)
        self.device = device
        self.checkpoint = checkpoint
        self.model_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  
        self.vocoder_optimizer = torch.optim.Adam(vocoder.parameters(), lr=0.001)  
        self.spectro_loss_func = nn.L1Loss()
        self.vocoder_loss_func = nn.CrossEntropyLoss()

    def train_step(self, entry):
        # ignore text
        _, audios, spectros = entry
        audios = audios.to(self.device)
        spectros = spectros.to(self.device)
        y_pred_spectros = self.model(spectros)

        # fix model returning 128 instead of 129 time steps in the spectrogram
        extra_timestep = y_pred_spectros[:,:,-1].unsqueeze(2)
        y_pred_spectros = torch.cat((y_pred_spectros, extra_timestep), dim=2)

        y_pred_spectros_temp = y_pred_spectros # store for later return for debug
        spectro_loss = self.spectro_loss_func(spectros, y_pred_spectros)
        
        # mels = y_pred_spectros.transpose(0,1)
        y_pred_spectros = torch.cat((y_pred_spectros, extra_timestep), dim=2)
        y_pred_spectros = torch.cat((y_pred_spectros, extra_timestep), dim=2)
        y_pred_spectros = torch.cat((y_pred_spectros, extra_timestep), dim=2)
        
        x = audios
        mels = y_pred_spectros
        y_pred_wavs = self.vocoder( x, mels )
        y_pred_wavs = y_pred_wavs.amax(dim=-1) # collapse 512 channel wav into 1 channel

        # this 0.001 ratio is not mentioned in the paper
        vocoder_loss = 0.001 * self.vocoder_loss_func(audios, y_pred_wavs)

        total_loss = spectro_loss + vocoder_loss


        self.model_optimizer.zero_grad()
        self.vocoder_optimizer.zero_grad()
        total_loss.backward()
        self.model_optimizer.step()
        self.vocoder_optimizer.step()

        ## free tensor from gpu memory?
        ## not sure if this works
        # del y_pred_spectro

        # last two values used for debugging only
        return total_loss, spectro_loss, vocoder_loss, y_pred_spectros_temp, y_pred_wavs


    def train(
        self,
        dataset=None,
        epochs=10,
        save_every_n=10,
        log_every_n=1,
        batch_size=5,
        run_name="runs/run1"
    ):
        if dataset is None:
            dataset = get_dataset()

        print(f"Beginning training {epochs} epochs, logging every {log_every_n}, saving every {save_every_n}.")
        
        import math
        total_batches = math.ceil(len(dataset)/batch_size)

        import torch.utils.tensorboard
        writer = torch.utils.tensorboard.SummaryWriter(run_name)

        for epoch in range(epochs):
            for i, entry in enumerate(dataset.batched(batch_size)):
                losses = self.train_step(entry)[:3] # ignore predicted values
                if (i+1) % log_every_n == 0:
                    Trainer.show_loss(epoch, f"{i}/{total_batches}", *losses)
                    time=epoch*total_batches + (i+1)*batch_size
                    writer.add_scalar('total loss', losses[0], time)
                    writer.add_scalar('spectro loss', losses[1], time)
                    writer.add_scalar('wav loss', losses[2], time)
                if (i+1) % save_every_n == 0:
                    self.checkpoint += (i+1)
                    self.save()
    
    def save(self):
        print(f"Saving checkpoint {self.checkpoint}")
        Trainer.save_as(self.model, f"model{self.checkpoint}.pth")
        Trainer.save_as(self.vocoder, f"model{self.checkpoint}_vocoder.pth")
        print("Saved")

    @classmethod
    def show_loss(self, epoch, i, total_loss, spectro_loss, vocoder_loss):
        print(
            f"epoch={epoch}",
            f"i={i}",
            f"total loss={total_loss.detach().numpy():.5f}",
            f"spectro loss={spectro_loss.detach().numpy():.5f}",
            f"vocoder loss={vocoder_loss.detach().numpy():.5f}",
        )

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
