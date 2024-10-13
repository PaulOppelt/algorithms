import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class data(Dataset):
    def __init__(self, features):
        self.features = features
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        x = self.features[idx]
        return torch.nn.functional.normalize(x).unsqueeze(0)

class encoder(nn.Module):
    def __init__(self,channels,latent_dim):
        super().__init__()
        self.l1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.l2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.mu = nn.Linear(16*7*7, latent_dim)
        self.sig = nn.Linear(16*7*7, latent_dim)

    def forward(self, x, batch_size):
        x = F.relu(self.l1(x))
        x = self.pool(x)
        x = F.relu(self.l2(x))
        x = self.pool(x)
        x = torch.flatten(x,1)
        mu, sig = self.mu(x), self.sig(x)
        return mu, sig
    
class decoder(nn.Module):
    def __init__(self,channels,latent_dim):
        super().__init__()
        self.l1 = nn.Linear(latent_dim,16*7*7)
        self.l2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1)
        self.l3 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
    def forward(self, x, batch_size):
        x = self.l1(x) 
        x = x.reshape(batch_size, 16, 7, 7)
        x = self.upsample(x)
        x = F.relu(self.l2(x))
        x = self.upsample(x)
        x = self.l3(x)
        return x

class image_vae(nn.Module):
    def __init__(self, latent_dim, channels):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = encoder(channels,latent_dim)
        self.decoder = decoder(channels,latent_dim)
        self.output = nn.Sigmoid()

    def forward(self,x, batch_size):
        mu, sig = self.encoder(x, batch_size)
        sig = torch.exp(0.5 * sig)
        x = mu + sig*torch.rand_like(sig)
        x = self.decoder(x, batch_size)
        x = self.output(x)
        return x, mu, sig
    
def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + KLD
