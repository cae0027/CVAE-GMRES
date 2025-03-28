# === Import Section ===
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import random

# Local imports
from architecture import CVAE
from trainHelpers import final_loss, fit, validate, device
from dataLoader import genDataloaders
# === Reproducibility ===
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# === Data Config ===
coarse_dim = 100
fine_dim = 2001
batch_size = 30

# === Load data from pre-saved .npz ===
train_loader, val_loader, test_loader = genDataloaders(
    path="fenics_output.npz",
    batch_size=batch_size
)

# === Training ===
def run_train(coarse_dim=coarse_dim, fine_dim=fine_dim, epochs=10, lr=1e-6, no_layers=4):
    model = CVAE(coarse_dim=coarse_dim, fine_dim=fine_dim, no_layers=no_layers).to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses, val_rmses = [], []

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss = fit(model, train_loader, optimizer, criterion)
        val_rmse = validate(model, val_loader, criterion)
        train_losses.append(train_loss)
        val_rmses.append(val_rmse)

        print(f"Train Loss: {train_loss:.6f} | Val RMSE: {val_rmse:.6f}")

        if np.isnan(val_rmse):
            model.isnan = True
            print("Model diverged (NaNs).")
            break
        else:
            model.isnan = False

    # Plot loss curves
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_rmses, label='Val RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('CVAE Training')
    plt.legend()
    plt.show()

    return model


# === Testing ===
def run_test(model):
    if getattr(model, "isnan", False):
        print("Skipping test — model did not converge.")
        return float('nan'), float('nan')

    model.eval()
    errL2, errRMSE = [], []

    with torch.no_grad():
        for yc, yf in test_loader:
            yc = yc.to(device).unsqueeze(0)
            z = torch.randn((1, model.latent_dim)).to(device)
            z = torch.cat((z, yc), dim=1)
            y = model.decode(z).squeeze(0).cpu().numpy()
            yf = yf.numpy().flatten()

            rmse = np.sqrt(np.mean((y - yf) ** 2))
            l2_error = np.linalg.norm(y - yf) / np.linalg.norm(yf)

            errRMSE.append(rmse)
            errL2.append(l2_error)

    print(f"Avg L2 Error: {np.mean(errL2):.6f}")
    print(f"Avg RMSE:     {np.mean(errRMSE):.6f}")
    return np.mean(errRMSE), np.mean(errL2)


# === Run everything ===
if __name__ == "__main__":
    model = run_train()
    run_test(model)
