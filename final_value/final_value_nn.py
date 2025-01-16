import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np

# Custom Dataset Class
class XYZDataset(Dataset):
    def __init__(self, csv_file):
        # Load the dataset from the CSV file
        data = pd.read_csv(csv_file)
        self.inputs = data[['X', 'Y']].values.astype(np.float32)  # X, Y as input
        self.targets = data['Z'].values.astype(np.float32)  # Z as target

        # Filter out rows with NaN values in Z
        valid_indices = ~np.isnan(self.targets)
        self.inputs = self.inputs[valid_indices]
        self.targets = self.targets[valid_indices]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)
        z = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, z

# Feedforward Neural Network
class FeedforwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# Training Function
def train_model(model, dataloader, criterion, optimizer, scheduler, num_epochs):
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in dataloader:
            inputs, targets = batch
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()  # Squeeze to match target shape
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Step the scheduler at the end of each epoch
        scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

# Main Program
if __name__ == "__main__":
    # Load dataset
    dataset = XYZDataset(csv_file='dataset.csv')
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Model parameters
    input_dim = 2  # X and Y
    hidden_dim = 128
    output_dim = 1  # Z

    # Initialize model, loss, optimizer, and scheduler
    model = FeedforwardNN(input_dim, hidden_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)  # Decay LR every 10 epochs by a factor of 0.5

    # Train the model
    num_epochs = 500000
    train_model(model, dataloader, criterion, optimizer, scheduler, num_epochs)

    # Save the trained model
    torch.save(model.state_dict(), 'feedforward_nn_with_scheduler.pth')
    print("Model training complete and saved as 'feedforward_nn_with_scheduler.pth'.")
