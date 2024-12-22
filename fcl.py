from comet_ml import Experiment
import sqlite3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

# Initialize Comet experiment
experiment = Experiment(
    api_key="6awmXJre5WWTHQN7EAGSS3PcT",
    project_name="cnn",
    workspace="melafernandez"
)

print("Comet experiment initialized.")

# 1. Connect to the SQLite database and extract data
conn = sqlite3.connect('shogi_board_states.db')
cursor = conn.cursor()
cursor.execute("SELECT winner, board_state FROM board_data LIMIT 200000")
db_data = cursor.fetchall()
conn.close()

print("Connected to SQLite database and fetched data.")

# 2. Preprocess the data
labels = [1 if row[0] == 'b' else 0 for row in db_data]
board_states = [[int(bit) for bit in row[1]] for row in db_data]

labels = torch.tensor(labels, dtype=torch.float32)
board_states = torch.tensor(board_states, dtype=torch.float32).view(-1, 1, 45, 12)

print("Data preprocessed.")

# 3. Create Dataset and DataLoader
dataset = data.TensorDataset(board_states, labels)

split_ratio = 0.8
split_idx = int(len(dataset) * split_ratio)
train_dataset, val_dataset = data.random_split(dataset, [split_idx, len(dataset) - split_idx])

train_loader = data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = data.DataLoader(val_dataset, batch_size=64, shuffle=False)

print("Dataset and DataLoader created.")

# 4. Construct the CNN classifier
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 45 * 12, 256)
        self.fc2 = nn.Linear(256, 128) #New Deep Layer
        self.fc3 = nn.Linear(128, 64) #New Deep Layer
        self.fc4 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        x = torch.squeeze(x)
        return x

print("CNN classifier constructed.")

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Move the model to the device
model = CNNClassifier().to(device)

# 5. Training setup
learning_rate = 0.001
epochs = 20

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

hyper_params = {
    "learning_rate": learning_rate,
    "epochs": epochs,
    "batch_size": 64
}
experiment.log_parameters(hyper_params)

print("Training setup complete.")

# 6. Training loop
for epoch in range(epochs):
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    val_losses = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            val_outputs = model(inputs)
            val_loss = criterion(val_outputs, targets)
            val_losses.append(val_loss.item())

    avg_val_loss = sum(val_losses) / len(val_losses)
    experiment.log_metric("train_loss", loss.item(), epoch=epoch)
    experiment.log_metric("val_loss", avg_val_loss, epoch=epoch)
    print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {loss.item()}, Validation Loss: {avg_val_loss}")

print("Training complete!")

# Log the model
log_model(experiment, model, model_name="ShogiCNN")

print("Model logged.")

