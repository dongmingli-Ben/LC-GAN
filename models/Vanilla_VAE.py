import torch.nn as nn
import torch

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 256, 5, 2, 1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 5, 2, 1),
            nn.ReLU(),
            nn.Conv2d(512, 1024, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(1024, 2048, 3, 2, 1),
            nn.ReLU(),
        )
        self.linear = nn.Linear(2048*4*4, 2048)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        mu = x[:, :1024]
        std = self.softplus(x[:, 1024:])
        return mu, std

class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(1024, 2048*4*4)
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, 3, 2, 1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, 3, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 5, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 3, 5, 2, 1, output_padding=1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        z = self.linear(z)
        z = z.view(z.size(0), 2048, 4, 4)
        z = self.layers(z)
        z = self.sigmoid(z)
        return z


class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.initialize()

    def initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, x):
        mu, std = self.encoder(x)
        return mu, std

    def decode(self, z):
        return self.decoder(z)

    @staticmethod
    def sample(mu, std):
        ep = torch.randn_like(std).to(mu.device)
        return mu + ep*std

    def forward(self, x):
        """return (mu, log_var, reconstructed x)"""
        mu, std = self.encode(x)
        z = self.sample(mu, std)
        y = self.decode(z)
        return mu, std, y