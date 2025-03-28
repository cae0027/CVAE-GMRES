import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from architecture import CVAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def final_loss(mse_loss, mu, logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
    return mse_loss + KLD

def fit(model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for uf, qc in dataloader:
        uf = uf.to(device).view(uf.size(0), -1)
        qc = qc.to(device).view(qc.size(0), -1)
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(uf, qc)
        mse_loss = criterion(reconstruction, uf)
        loss = final_loss(mse_loss, mu, logvar)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    return running_loss / len(dataloader.dataset)


def validate(model, dataloader, criterion):
    model.eval()
    total_sq_error = 0.0
    n_samples = 0

    with torch.no_grad():
        for uf, qc in dataloader:
            uf = uf.to(device).view(uf.size(0), -1)
            qc = qc.to(device).view(qc.size(0), -1)
            reconstruction, _, _ = model(uf, qc)
            batch_error = ((reconstruction - uf) ** 2).sum().item()
            total_sq_error += batch_error
            n_samples += uf.numel()

    rmse = np.sqrt(total_sq_error / n_samples)
    return rmse


if __name__ == '__main__':
    # model = CVAE(
    #     coarse_dim=22,
    #     latent_dim=2,
    #     enc_widths=[28, 20, 14],
    #     dec_widths=[14, 20, 28],
    #     no_layers=3
    # ).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # criterion = torch.nn.MSELoss()
    # train_loss = fit(model, dataloaderf, optimizer, criterion)
    # val_loss, true_soln, pred_soln = validate(model, dataloaderf, criterion, plot=True)
    # print(train_loss, val_loss)
    # print(true_soln, pred_soln)
    # print(true_soln.shape, pred_soln.shape)  

    from torch.utils.data import DataLoader, TensorDataset
    import torch.optim as optim

    # Dummy dataset
    xf = torch.randn(64, 2001)
    xc = torch.randn(64, 22)
    dataset = TensorDataset(xf, xc)
    loader = DataLoader(dataset, batch_size=8)

    # Model and optimizer
    model = CVAE(coarse_dim=22, fine_dim=2001, no_layers=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    print(model)

    # Train/validate
    train_loss = fit(model, loader, optimizer, criterion)
    val_loss, true_soln, pred_soln = validate(model, loader, criterion, plot=True)

    print(f"Train loss: {train_loss:.4f}")
    print(f"Val loss: {val_loss:.4f}")
 