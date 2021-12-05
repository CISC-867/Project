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
        bits=9,
        pad=2,
        upsample_factors=(5, 5, 8),
        feat_dims=80,
        compute_dims=128,
        res_out_dims=128,
        res_blocks=10,
        hop_length=200,
        sample_rate=16000,
        mode="RAW",
    )

def get_spectrogram_model():
    import model.model
    return model.model.SpectrogramModel()

def get_dataset():
    import data.VCTK
    return data.VCTK.VCTKDataset("VCTK-Corpus-smaller/")

# F.cross_entropy keeps giving negative numbers
# so this hack should fix that I guess.
class spicy_loss:
    def __init__(self, device) -> None:
        self.device = device
    def __call__(self, y_hat, y) -> float:
        return (y - y_hat).abs().sum()
        # offset = torch.ones(y_hat.shape) * 1.2
        # offset = offset.to(self.device)
        # y_hat, y = y_hat + offset, y + offset
        # return F.cross_entropy(y_hat, y)

class Trainer:
    def __init__(self,
        model=None,
        vocoder=None,
        device=None,
        checkpoint=None,
        checkpoint_dir="checkpoints",
        model_checkpoint_pattern="model{}.pth",
        vocoder_checkpoint_pattern="model{}_vocoder.pth",
        load_from_checkpoint=False,
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
            Trainer.load_from(model, model_checkpoint_pattern.format(checkpoint), dir=checkpoint_dir)
            Trainer.load_from(vocoder, vocoder_checkpoint_pattern.format(checkpoint), dir=checkpoint_dir)
            print("Models loaded")

        if checkpoint is None:
            checkpoint = 0
        self.model = model.to(device)
        self.vocoder = vocoder.to(device)
        self.device = device
        self.checkpoint = checkpoint
        self.checkpoint_dir = checkpoint_dir
        self.model_checkpoint_pattern = model_checkpoint_pattern
        self.vocoder_checkpoint_pattern = vocoder_checkpoint_pattern
        self.model_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  
        self.vocoder_optimizer = torch.optim.Adam(vocoder.parameters(), lr=0.001)  
        self.spectro_loss_func = nn.L1Loss()
        # self.vocoder_loss_func = F.cross_entropy # this doesn't work right, giving negative values
        self.vocoder_loss_func = spicy_loss(device)

    def train_step(self, entry, eval=False):
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
        y_pred_wavs_temp = y_pred_wavs

        ## this is bad.
        # y_pred_wavs = y_pred_wavs.amax(dim=-1) # collapse 512 channel wav into 1 channel
        y_pred_wavs = y_pred_wavs[:,:,-1] # take the last guess as output

        if eval:
            return y_pred_spectros_temp, y_pred_wavs

        # this 0.001 ratio is not mentioned in the paper
        # vocoder_loss = 0.001 * self.vocoder_loss_func(audios, y_pred_wavs)
        vocoder_loss = 0.001 * self.vocoder_loss_func(y_pred_wavs, audios)

        total_loss = spectro_loss + vocoder_loss

        self.model_optimizer.zero_grad()
        self.vocoder_optimizer.zero_grad()
        total_loss.backward()
        self.model_optimizer.step()
        self.vocoder_optimizer.step()

        # last tuple used for debugging, inference time returns earlier above
        return total_loss, spectro_loss, vocoder_loss, (y_pred_spectros_temp, y_pred_wavs, y_pred_wavs_temp)


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

        print(f"Beginning training {epochs} epochs, logging every {log_every_n}, saving every {save_every_n}, batch size {batch_size}.")
        
        import math

        import torch.utils.tensorboard
        writer = torch.utils.tensorboard.SummaryWriter(run_name)

        cpu = torch.device("cpu")

        total_clips = 0
        for epoch in range(epochs):
            for i, entry in enumerate(dataset.batched(batch_size)):
                if epoch == 0: total_clips += batch_size
                self.checkpoint += 1
                total_loss, spectro_loss, vocoder_loss, _ = self.train_step(entry)
                total_loss = total_loss.detach().to(cpu).numpy()
                spectro_loss = spectro_loss.detach().to(cpu).numpy()
                vocoder_loss = vocoder_loss.detach().to(cpu).numpy()
                if (i+1) % log_every_n == 0:
                    Trainer.show_loss(
                        epoch,
                        f"{i}/{math.ceil(total_clips/batch_size)}",
                        total_loss,
                        spectro_loss,
                        vocoder_loss
                    )
                    time=epoch*total_clips + (i+1)*batch_size
                    writer.add_scalar('total_loss', total_loss, time)
                    writer.add_scalar('spectro_loss', spectro_loss, time)
                    writer.add_scalar('vocoder_loss', vocoder_loss, time)
                if (i+1) % save_every_n == 0:
                    self.save()
        self.checkpoint+=1
        self.save()
    
    def save(self):
        try: # ignore pipe errors when ctrl+c used
            print(f"Saving checkpoint {self.checkpoint}")
        except BrokenPipeError:
            pass
        Trainer.save_as(self.model, self.model_checkpoint_pattern.format(self.checkpoint), dir=self.checkpoint_dir)
        Trainer.save_as(self.vocoder, self.vocoder_checkpoint_pattern.format(self.checkpoint), dir=self.checkpoint_dir)
        try:
            print("Saved")
        except BrokenPipeError:
            pass
            

    @classmethod
    def show_loss(self, epoch, batch, total_loss, spectro_loss, vocoder_loss):
        print(
            f"epoch={epoch}",
            f"batch={batch}",
            f"total_loss={total_loss:.5f}",
            f"spectro_loss={spectro_loss:.5f}",
            f"vocoder_loss={vocoder_loss:.5f}",
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
