import torch
from torch.utils.data import Dataset, DataLoader

from glob import glob
from os.path import join, basename, splitext

class VCTKDataset(Dataset):
    def __init__(self, path) -> None:
        super().__init__()
        
        # load file paths
        text_file_paths = sorted(glob(join(path, "txt", "*", "*.txt")))
        audio_file_paths = sorted(glob(join(path, "wav48", "*", "*.wav")))
        self.data = []

        for text, audio in zip(text_file_paths, audio_file_paths):
            # sanity check
            assert basename(splitext(text)[0]) == basename(splitext(audio)[0])

            self.data += [(text, audio)]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
