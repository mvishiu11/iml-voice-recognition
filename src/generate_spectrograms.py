import os
import librosa
import librosa.display
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# Define directories
data_dir = os.path.join("..", "data", "raw")
processed_dir = os.path.join("..", "data", "processed")
if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)

# Load metadata
metadata = []
class_1_speakers = ['f1', 'f7', 'f8', 'm3', 'm6', 'm8']

for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith(".wav"):
            speaker_id = file.split('_')[0]
            label = 1 if speaker_id in class_1_speakers else 0
            metadata.append({
                'path': os.path.join(root, file),
                'label': label
            })

print(f"Found {len(metadata)} audio files")

# Convert metadata to DataFrame
metadata_df = pd.DataFrame(metadata)

# Function to load and split audio into segments
def load_and_split_audio(file_path, target_sr=16000, segment_length=5):
    # Load audio
    audio, sr = librosa.load(file_path, sr=target_sr)
    total_length = len(audio) / sr
    segments = []
    
    # Split audio into segments of length 'segment_length'
    for start in np.arange(0, total_length, segment_length):
        end = start + segment_length
        segment = audio[int(start * sr):int(end * sr)]
        # Pad if the segment is shorter
        if len(segment) < segment_length * sr:
            segment = np.pad(segment, (0, int(segment_length * sr) - len(segment)), mode='constant')
        segments.append(segment)
    return segments

# Function to generate and save spectrograms
def save_spectrogram(segment, sr, label, index):
    # Generate Mel Spectrogram
    S = librosa.feature.melspectrogram(y=segment, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    S_DB = librosa.power_to_db(S, ref=np.max)
    
    # Save spectrogram as an image
    plt.figure(figsize=(6, 4))
    librosa.display.specshow(S_DB, sr=sr, hop_length=512, x_axis='time', y_axis='mel', cmap='viridis')
    plt.axis('off')
    spectrogram_filename = f"spectrogram_{label}_{index}.png"
    spectrogram_path = os.path.join(processed_dir, spectrogram_filename)
    plt.savefig(spectrogram_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Process each audio file
sr = 16000
index = 0
for _, row in tqdm(metadata_df.iterrows(), desc="Processing Audio Files", total=len(metadata_df)):
    segments = load_and_split_audio(row['path'], target_sr=sr)
    for segment in segments:
        save_spectrogram(segment, sr, row['label'], index)
        index += 1

print(f"Saved {index} spectrograms to {processed_dir}")