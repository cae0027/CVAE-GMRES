# load_dataloaders.py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def genDataloaders(
    path='fenics_output.npz',
    batch_size=30,
    train_frac=0.8,
    val_frac=0.1
):
    data = np.load(path)
    A = data['A']
    qc = data['qc']
    m = A.shape[0]

    train_idx = int(m * train_frac)
    val_idx = int(m * (train_frac + val_frac))

    qc_train, qc_val, qc_test = qc[:train_idx], qc[train_idx:val_idx], qc[val_idx:]
    A_train, A_val, A_test = A[:train_idx], A[train_idx:val_idx], A[val_idx:]

    train_data = TensorDataset(torch.tensor(A_train, dtype=torch.float32), torch.tensor(qc_train, dtype=torch.float32))
    val_data = TensorDataset(torch.tensor(A_val, dtype=torch.float32), torch.tensor(qc_val, dtype=torch.float32))
    test_data = TensorDataset(torch.tensor(A_test, dtype=torch.float32), torch.tensor(qc_test, dtype=torch.float32))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_data

if __name__ == "__main__":
    train_loader, val_loader, test_loader = genDataloaders("fenics_output.npz", batch_size=2)

    for A_batch, qc_batch in train_loader:
        print("A shape:", A_batch.shape)
        print("qc shape:", qc_batch.shape)
        break
