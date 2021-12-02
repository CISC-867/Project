import torch
import os
import glob
import librosa

import tqdm.notebook as tqdm

class VCTKDataset(torch.utils.data.Dataset):
    def __init__(self, path) -> None:
        super().__init__()
        
        # load file paths
        text_file_paths = sorted(glob.glob(os.path.join(path, "txt", "*", "*.txt")))
        audio_file_paths = sorted(glob.glob(os.path.join(path, "wav48", "*", "*.wav")))
        self.data = []

        for text, audio in tqdm.tqdm(list(zip(text_file_paths, audio_file_paths))):
            # sanity check
            assert os.path.basename(os.path.splitext(text)[0]) == os.path.basename(os.path.splitext(audio)[0])

            audio = librosa.load(audio)

            self.data += [(text, audio)]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
