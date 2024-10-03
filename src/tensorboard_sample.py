import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data_loader import create_data_loader
from torch.utils.tensorboard import SummaryWriter

# Initialize TensorBoard writer
writer = SummaryWriter("runs/voice_recognition_experiment_1")


# Example: Model definition with final output layer for binary classification
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # We don't know the size for fc yet, so we initialize it after calculating it dynamically
        self.fc = None

    def forward(self, x):
        # Apply convolution and pooling
        x = self.pool(F.relu(self.conv1(x)))

        # Flatten the tensor but preserve the batch size
        x = x.view(x.size(0), -1)

        # Dynamically calculate the number of features and initialize the fc layer
        if self.fc is None:
            num_features = x.size(
                1
            )  # Calculate the number of features from the flattened tensor
            self.fc = nn.Linear(
                num_features, 2
            )  # Output 2 classes (binary classification)

        x = self.fc(x)
        return x

    def initialize_fc(self, input_size):
        # Perform a forward pass with a dummy input to determine the size of the fully connected layer
        dummy_input = torch.zeros(1, 1, input_size, input_size)
        dummy_output = self.pool(F.relu(self.conv1(dummy_input)))
        num_features = dummy_output.view(1, -1).size(1)
        self.fc = nn.Linear(num_features, 2)  # Output for 2 classes


# Dummy loss and optimizer
model = SimpleCNN()
model.initialize_fc(64)  # Assume 64x64 spectrogram input size
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Path to data
data_dir = os.path.join("data", "raw")

# Create data loader
train_loader = create_data_loader(data_dir, batch_size=32, shuffle=True)

# Training loop
for epoch in range(5):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation and optimization step
        loss.backward()
        optimizer.step()

        # Log loss to TensorBoard
        running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {loss.item()}")
        if i % 100 == 99:
            writer.add_scalar(
                "training loss", running_loss / 100, epoch * len(train_loader) + i
            )
            running_loss = 0.0

# Close the writer
writer.close()
