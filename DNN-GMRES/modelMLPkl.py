import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json

# --------------------------
# Detect device
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# q(x, ξ) function
# --------------------------
def q_function(x, xi, d=0.0, c=5.0, beta=0.1):
    return d + np.exp(c * np.cos(2 * np.pi * x / beta + xi))

# --------------------------
# PyTorch Dataset
# --------------------------
class LocalStiffnessDataset(Dataset):
    def __init__(self, json_path, num_pts=20):
        super().__init__()
        with open(json_path, "r") as f:
            self.data = json.load(f)

        self.num_pts = num_pts
        self.samples = []

        for sample in self.data:
            seed = sample["seed"]
            for elem in sample["elements"]:
                x0, x1 = elem["interval"]
                q_vals = elem["q_vals"]
                stiffness = elem["stiffness"]
                self.samples.append((seed, x0, x1, q_vals, stiffness))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seed, x0, x1, q_vals, stiffness = self.samples[idx]
        return {
            "q_vals": torch.tensor(q_vals, dtype=torch.float32),
            "stiffness": torch.tensor(stiffness, dtype=torch.float32),
            "interval": torch.tensor([x0, x1], dtype=torch.float32),
            "seed": torch.tensor(seed, dtype=torch.float32)
        }

# --------------------------
# MLP Model
# --------------------------
class LocalStiffnessMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, x):
        return self.net(x)

# --------------------------
# Training Function
# --------------------------
def train_model(model, dataloader, epochs=10, lr=1e-3):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            q_input = batch["q_vals"].to(device)
            target = batch["stiffness"].to(device)

            pred = model(q_input)
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {epoch_loss / len(dataloader):.6f}")

# --------------------------
# Stiffness Reconstruction
# --------------------------
def reconstruct_stiffness(model, dataset, sample_idx, num_pts=20):
    model.eval()
    model.to(device)

    # Get the sample
    sample = dataset.data[sample_idx]
    seed = sample["seed"]
    element_data = sample["elements"]
    nf = len(element_data)

    A_pred = np.zeros((nf + 1, nf + 1))
    A_true = np.zeros((nf + 1, nf + 1))

    with torch.no_grad():
        for i, elem in enumerate(element_data):
            x0, x1 = elem["interval"]
            true_stiffness = np.array(elem["stiffness"])

            # --- True matrix assembly ---
            A_true[i, i]     += true_stiffness[0]
            A_true[i, i+1]   += true_stiffness[1]
            A_true[i+1, i]   += true_stiffness[1]
            A_true[i+1, i+1] += true_stiffness[2]

            # --- Predicted matrix assembly ---
            # x_grid = np.linspace(x0, x1, num_pts)
            # q_vals = q_function(x_grid, xi, d=dataset.d, c=dataset.c, beta=dataset.beta)
            q_vals = elem["q_vals"] 
            q_tensor = torch.tensor(q_vals, dtype=torch.float32, device=device).unsqueeze(0)
            pred = model(q_tensor).squeeze().cpu().numpy()

            A_pred[i, i]     += pred[0]
            A_pred[i, i+1]   += pred[1]
            A_pred[i+1, i]   += pred[1]
            A_pred[i+1, i+1] += pred[2]

    return A_pred, A_true

