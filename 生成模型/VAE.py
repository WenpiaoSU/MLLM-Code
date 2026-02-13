# 变分自编码器VAE
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    # 编码
    def encode(self, x):
        h = nn.ReLU(self.fc1(x))
        mu = self.fc_mu(h)   # 均值
        logvar = self.fc_logvar(h)  # 方差
        return mu, logvar

    # 重参数化技巧
    def reparameterize(self, mu, logvar):
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*std.shape)
        z = mu + std * eps
        return z

    # 解码
    def decode(self, z):
        h = nn.ReLU(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

def vae_loss(x, x_recon, mu, logvar):
    """ELBO 损失"""
    # 重构损失（负对数似然）
    recon_loss = np.sum((x - x_recon) ** 2)
    # KL散度
    kl_loss = -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar))
    return recon_loss + kl_loss

