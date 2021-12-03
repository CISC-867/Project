import torch
import os
import glob
import librosa, librosa.display, librosa.util
import numpy as np
import math

class VCTKDataset(torch.utils.data.Dataset):
    def __init__(self, path) -> None:
        super().__init__()
        self.sample_rate=16000
        self.win_length=800
        self.hop_length=200
        self.n_mels=80
        self.clip_length=int(self.sample_rate*1.6)
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
        # trim silence
        audio, _ = librosa.effects.trim(audio)
        # convert from 48khz to 16khz for efficiency
        audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        # pad audio to clip length
        full_clips = math.ceil(len(audio)/self.clip_length)
        full_length = full_clips * self.clip_length
        audio = librosa.util.fix_length(audio, full_length)

        clips = []
        for clip in audio.reshape((-1, self.clip_length)):
            # convert to spectrogram
            # https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html
            # https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0
            spectro = librosa.feature.melspectrogram(
                y=clip,
                sr=self.sample_rate,
                win_length=self.win_length,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
            )
            clips.append(spectro)
        
        return text, clips
        # return text, audio

    def __len__(self):
        return len(self.data)

    def show(self, entry):
        # https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0
        import matplotlib.pyplot as plt
        text, spectros = entry
        for v in spectros:
            fig, ax = plt.subplots()
            img = librosa.display.specshow(
                librosa.power_to_db(v, ref=np.max),
                sr=self.sample_rate,
                hop_length=self.hop_length,
                x_axis="time",
                y_axis="mel",
                ax=ax
            )
            plt.colorbar(img, ax=ax, format="%+2.0f dB")


