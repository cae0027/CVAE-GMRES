from modelMLP import LocalStiffnessDataset, LocalStiffnessMLP, train_model, reconstruct_stiffness
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# from testAoAlMatrices import testAoAlMatrices
from toy import testAoAlMatrices

# Load dataset
json_path = "fem_local_stiffness_data.json"
dataset = LocalStiffnessDataset(json_path, num_pts=2)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Create model
model = LocalStiffnessMLP(input_dim=2)


# Train
train_model(model, loader, epochs=5)

# Test: Reconstruct stiffness matrix from model
x_nodes = np.linspace(-1, 1, 1001)
# xi_test = 0.78
# After training...
sample_idx = 0  # Any valid index from 0 to len(dataset)-1
Al, Ao = reconstruct_stiffness(model, dataset, sample_idx)
testAoAlMatrices(Al, Ao)

plt.plot(np.abs(np.diag(Ao)- np.diag(Al)))
plt.title("Absolute Difference: Ao_diag - Al_diag")
plt.xlabel("Index")
plt.ylabel("Difference")
plt.grid()
plt.show()



# Comparison Plot
plt.plot(np.diag(Ao), label="True A_ii")
plt.plot(np.diag(Al), '--', label="Predicted A_ii")
plt.legend()
plt.title("Diagonal of Stiffness Matrix: Truth vs Prediction")
plt.xlabel("Index")
plt.ylabel("A_ii value")
plt.grid(True)
plt.show()

