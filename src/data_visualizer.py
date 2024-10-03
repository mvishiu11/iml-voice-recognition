import os

import matplotlib.pyplot as plt
import torchaudio.transforms as T
from data_loader import create_data_loader

# Path to data
data_dir = os.path.join("data", "raw")

# Create DataLoader
train_loader = create_data_loader(data_dir, batch_size=4, shuffle=True)

# Get one batch from the DataLoader
data_iter = iter(train_loader)
spectrograms, labels = next(data_iter)

# Apply amplitude-to-decibel conversion to visualize properly
amplitude_to_db = T.AmplitudeToDB()


# Function to visualize a single spectrogram
def show_spectrogram(spectrogram, label):
    # Convert to decibel scale for better visualization
    spectrogram = amplitude_to_db(spectrogram)

    spectrogram = spectrogram.squeeze(
        0
    ).numpy()  # Remove channel dimension and convert to NumPy
    plt.imshow(
        spectrogram, cmap="viridis", origin="lower"
    )  # Display the spectrogram as an image
    plt.title(f"Label: {label}")
    plt.colorbar()
    plt.show()


# Loop through the batch and display each spectrogram
for i in range(spectrograms.size(0)):
    show_spectrogram(spectrograms[i], labels[i].item())
