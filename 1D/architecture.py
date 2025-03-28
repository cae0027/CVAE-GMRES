import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CVAE(nn.Module):
    def __init__(self,
                 coarse_dim=50,
                 fine_dim=2001,
                 latent_dim=22,
                 enc_widths=None,
                 dec_widths=None,
                 no_layers=4):
        super().__init__()
        self.coarse_dim = coarse_dim
        self.fine_dim = fine_dim
        self.latent_dim = latent_dim

        # Auto-generate encoder widths if not provided
        if enc_widths is None:
            enc_widths = self._generate_widths(start=fine_dim, end=latent_dim * 2, no_layers=no_layers)
            # increase encoder input by 1 to match fine dim
            enc_widths[0] += 1
        else:
            enc_widths = [fine_dim] + enc_widths + [latent_dim * 2]

        # Auto-generate decoder widths if not provided
        if dec_widths is None:
            dec_widths = self._generate_widths(start=latent_dim + coarse_dim, end=fine_dim, no_layers=no_layers)
            # increase decoder output by 1 to match fine dim
            dec_widths[-1] += 1
        else:
            dec_input = latent_dim + coarse_dim
            dec_widths = [dec_input] + dec_widths + [fine_dim]

        # Build encoder and decoder
        self.encoder_layers = nn.ModuleList([
            nn.Linear(enc_widths[i], enc_widths[i+1])
            for i in range(len(enc_widths) - 1)
            ])


        self.decoder_layers = nn.ModuleList([
            nn.Linear(dec_widths[i], dec_widths[i+1])
            for i in range(len(dec_widths) - 1)
            ])

    def _generate_widths(self, start, end, no_layers, mode='down'):
        """
        Generates hidden layer sizes including `start` and `end`, evenly spaced logarithmically.
        """
        widths = np.logspace(np.log10(start), np.log10(end), num=no_layers + 1, base=10).astype(int).tolist()
        return widths


    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, xf):
        x = xf
        for layer in self.encoder_layers[:-1]:
            # print("Shape of x:", x.shape)
            x = F.relu(layer(x))
            # print("Shape of x after relu:", x.shape)
        x = self.encoder_layers[-1](x)
        # print("Shape of x after last layer:", x.shape)
        mu, log_var = x.chunk(2, dim=-1)
        # print("mu shape:", mu.shape, "log_var shape:", log_var.shape)
        # print norm of log_var
        print("Norm of log_var:", torch.norm(log_var))
        return mu, log_var

    def decode(self, z_cond):
        x = z_cond
        for layer in self.decoder_layers[:-1]:
            # print("Shape of x decoder:", x.shape)
            x = F.relu(layer(x))
            # print("Shape of x after relu decoder:", x.shape)
        x = self.decoder_layers[-1](x)
        # print("Shape of x after decoder last layer:", x.shape)
        return x

    def forward(self, xf, xc):
        mu, log_var = self.encode(xf)
        z = self.reparameterize(mu, log_var)
        z_cond = torch.cat([z, xc], dim=1)
        reconstruction = self.decode(z_cond)
        return reconstruction, mu, log_var

if __name__ == '__main__':
    model = CVAE(
    coarse_dim=22,
    fine_dim=2001,
    latent_dim=32,
    enc_widths=[1024, 512, 128],     # encoder widths
    dec_widths=[128, 512, 1024]      # decoder widths
    )

    print(model)