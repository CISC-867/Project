import torch
import os
import glob
import librosa
import numpy as np

class VCTKDataset(torch.utils.data.Dataset):
    def __init__(self, path) -> None:
        super().__init__()
        self.sample_rate=16000
        self.win_length=800
        self.hop_length=200
        self.n_mels=80
        
        # load file paths
        text_file_paths = sorted(glob.glob(os.path.join(path, "txt", "*", "*.txt")))
        audio_file_paths = sorted(glob.glob(os.path.join(path, "wav48", "*", "*.wav")))
        self.data = []

        for text, audio in zip(text_file_paths, audio_file_paths):
            # sanity check
            assert os.path.basename(os.path.splitext(text)[0]) == os.path.basename(os.path.splitext(audio)[0])

            self.data += [(text, audio)]

    def __getitem__(self, index):
        text, audio = self.data[index]
        
        # read transcript
        with open(text, "r") as handle:
            text = "".join([line for line in handle.readlines()]).strip()

        # load audio, also get sample rate
        audio, sr = librosa.load(audio)

        # convert from 48khz to 16khz for efficiency
        audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

        # convert to spectrogram
        # https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html
        spectro = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
        return text, spectro
        # return text, audio

    def __len__(self):
        return len(self.data)

    def show(self, entry):
        import matplotlib.pyplot as plt
        text, mel_spectro = entry
        librosa.display.specshow(
            librosa.power_to_db(mel_spectro, ref=np.max),
            sr=self.sample_rate,
            hop_length=self.hop_length,
            x_axis="time",
            y_axis="mel"
        )
        plt.colorbar(format="%+2.0f dB")


