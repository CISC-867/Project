import torch
import os
import glob
import librosa
import numpy as np

def download_dataset(version, save_path):
    if version == "Full":
        url = "https://queensuca-my.sharepoint.com/:u:/g/personal/16drp4_queensu_ca/EcnCAgHdAwBJmWk9vNcpiKgB5Rhi6GNtSchx6zGEhWe-fw?download=1"
        filename = "VCTK-Corpus.tar.gz"
    elif version == "Small":
        url = "https://queensuca-my.sharepoint.com/:u:/g/personal/16drp4_queensu_ca/Ebz_WoXK4T9KjbJq_SFCsZQB94PqsKOnQ9HwB5olpGeQIw?download=1"
        filename = "VCTK-Corpus-small.7z"
    elif version == "Smaller":
        url = "https://queensuca-my.sharepoint.com/:u:/g/personal/16drp4_queensu_ca/EfRSVbO-ohRFlUFhA9lx4qsBVl4s5FU676hF9uTRSAjOgg?download=1"
        filename = "VCTK-Corpus-smaller.tar.gz"
    
    archive_path = os.path.join(save_path, filename)
    base, ext = os.path.splitext(os.path.basename(archive_path))
    destination = os.path.join(os.path.dirname(archive_path), base)
    if os.path.exists(destination):
        print("already extracted")
        return

    # import urllib.request
    # import requests.utils
    # print("got",url)
    # url = requests.utils.requote_uri(url)
    import requests
    if not os.path.exists(archive_path):
        print(f"downloading archive")
        with open(archive_path, "wb") as out:
            req = requests.get(url)
            out.write(req.content)
    else:
        print("archive already downloaded")
        # urllib.request.urlretrieve(url, archive_path)


    print("extracting archive")
    import shutil
    shutil.unpack_archive(archive_path, destination)
    # import tarfile
    # if ext == ".tar.gz":
    #     print("tar.gz")
    # elif ext == ".tar":
    #     print("tar")
    # elif ext == ".7z":
    #     print("7z")

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
        # trim silence
        audio, _ = librosa.effects.trim(audio)
        # convert from 48khz to 16khz for efficiency
        audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        # convert to spectrogram
        # https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html
        # https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0
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
        # https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0
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


