import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch import nn

class AudioDataset(Dataset):
    def __init__(self,
                 directory=None,
                 sr=44100,
                 seg_len = 262144,
                 stereo=False,
                 train=True,
                 ds_percent=50,
                 ):
        super(AudioDataset, self).__init__()
        self.sr = sr
        self.directory = directory
        self.seg_len = seg_len
        self.train = train
        self.stereo = stereo
        self.ds_percent = ds_percent

        files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(".wav")]
        ds_idx = int(len(files) * (ds_percent / 100)) # Dataset index which divides files for training and testing

        if self.train:
            self.files = files[:ds_idx]  # First part of the dataset for training
        else:
            self.files = files[ds_idx:]  # Remaining files for testing

    def load_segment(self, audio_file):
        audio, sr = torchaudio.load(audio_file, normalize=True)
        # Min max normalization to ensure the dynamic range is within the values of [-1, 1]:
        audio = (audio - audio.min()) / (audio.max() - audio.min()) * 2 - 1
        if sr != self.sr:
            raise ValueError(f"Expected sample rate of {self.sr} but got {sr}")
        if self.stereo:
            audio = audio.mean(dim=0, keepdim=True)
        if audio.size(1) < self.seg_len:
            audio = nn.functional.pad(audio, (0, self.seg_len - audio.size(1)))
        elif audio.size(1) > self.seg_len:
            idx = torch.randint(0, audio.size(1) - self.seg_len, (1,)).item()
            audio = audio[:, idx:idx + self.seg_len]
        return audio

    def print_params(self):
        print(f"Path:                   {self.directory}")
        print(f"Sample rate:            {self.sr}")
        print(f"Segment length:         {self.seg_len}")
        print(f"Stereo:                 {self.stereo}")
        print(f"Number of files:        {len(self.files)}")
        print(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return self.load_segment(self.files[index])

if __name__ == "__main__":
    dataset = AudioDataset(directory="./IDMT-SMT-GUITAR/", sr=44100, seg_len=262144)
    dataset.print_params()
    dataloader = DataLoader(dataset, batch_size=4)
    print(next(iter(dataloader)).shape)
