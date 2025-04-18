from modelMLPkl import LocalStiffnessDataset, LocalStiffnessMLP, train_model, reconstruct_stiffness
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json


# Load dataset
json_path = "fem_stiffness_data_kl.json"
dataset = LocalStiffnessDataset(json_path, num_pts=20)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Create model
model = LocalStiffnessMLP(input_dim=20)

# Train
train_model(model, loader, epochs=10)

# Test: Reconstruct stiffness matrix from model
x_nodes = np.linspace(-1, 1, 1001)
# xi_test = 0.78
# After training...
sample_idx = 0  # Any valid index from 0 to len(dataset)-1
A_hat, A_true = reconstruct_stiffness(model, dataset, sample_idx)

# Comparison Plot
plt.plot(np.diag(A_true), label="True A_ii")
plt.plot(np.diag(A_hat), '--', label="Predicted A_ii")
plt.legend()
plt.title("Diagonal of Stiffness Matrix: Truth vs Prediction")
plt.xlabel("Index")
plt.ylabel("A_ii value")
plt.grid(True)
plt.show()

