import os

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchaudio.transforms import AmplitudeToDB
from torchaudio.transforms import MelSpectrogram


class DAPSDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_transform=None):
        """
        Args:
            data_dir (str): Path to the directory containing the dataset.
            transform (callable, optional): Optional transform to apply to the data (e.g., spectrogram).
            target_transform (callable, optional): Optional transform to apply to the targets.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.file_paths, self.labels = self.load_file_paths()

    def load_file_paths(self):
        """
        Loads file paths and corresponding labels from the dataset directory.

        Returns:
            file_paths (list): List of file paths for the audio files.
            labels (list): List of labels (0 or 1) corresponding to each audio file.
        """
        file_paths = []
        labels = []

        class_1_speakers = [
            "F1",
            "F7",
            "F8",
            "M3",
            "M6",
            "M8",
        ]  # List of class 1 speakers
        print(f"Loading data from {self.data_dir}")
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(".wav"):
                    speaker_id = os.path.basename(root).split("_")[0]
                    file_paths.append(os.path.join(root, file))
                    # Assign label 1 for allowed speakers (class 1), 0 for not allowed speakers (class 0)
                    label = 1 if speaker_id in class_1_speakers else 0
                    labels.append(label)

        print(f"Loaded {len(file_paths)} files")
        return file_paths, labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            sample (tuple): Tuple of (spectrogram, label) where spectrogram is the mel spectrogram of the audio.
        """
        audio_path = self.file_paths[idx]
        label = self.labels[idx]

        # Load the audio file
        waveform, sample_rate = torchaudio.load(audio_path)

        # Apply transforms (e.g., convert to MelSpectrogram)
        if self.transform:
            waveform = self.transform(waveform)

        if self.target_transform:
            label = self.target_transform(label)

        return waveform, label


def pad_sequence(batch):
    """
    Pads the spectrograms in the batch so that all have the same size.
    """
    spectrograms = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])

    # Find max width in the batch
    max_len = max([s.size(2) for s in spectrograms])

    # Pad all spectrograms to the same width
    padded_spectrograms = [F.pad(s, (0, max_len - s.size(2))) for s in spectrograms]

    # Stack the spectrograms into a single tensor
    padded_spectrograms = torch.stack(padded_spectrograms)

    return padded_spectrograms, labels


def create_data_loader(data_dir, batch_size=32, shuffle=True):
    """
    Creates a DataLoader for the DAPS dataset.

    Args:
        data_dir (str): Path to the dataset directory.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        data_loader (DataLoader): PyTorch DataLoader with preprocessed audio data.
    """
    # Define the transforms: convert to MelSpectrogram and scale the amplitude
    mel_spectrogram_transform = torch.nn.Sequential(
        MelSpectrogram(
            sample_rate=16000, n_fft=400, n_mels=64
        ),  # Reduce n_mels or increase n_fft
        AmplitudeToDB(),
    )

    # Create dataset and dataloader
    dataset = DAPSDataset(data_dir, transform=mel_spectrogram_transform)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_sequence
    )

    return data_loader


# Example usage:
if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(os.getcwd), "data", "raw")
    batch_size = 32

    # Create the data loader
    data_loader = create_data_loader(data_dir, batch_size)

    # Iterate over the data loader
    for batch_idx, (spectrograms, labels) in enumerate(data_loader):
        print(f"Batch {batch_idx + 1}")
        print(f"Spectrograms shape: {spectrograms.shape}")
        print(f"Labels: {labels}")
