import torch.nn as nn
import torch.nn.functional as F

from src.common.model import RecoverableModel


class SAE(RecoverableModel):
    def __init__(self, d_in, d_latent, sae_name: str = "sae_1_mlp.pt"):
        super().__init__()
        self.enc = nn.Linear(d_in, d_latent, bias=False)
        self.dec = nn.Linear(d_latent, d_in, bias=False)
        
        nn.init.kaiming_uniform_(self.enc.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.dec.weight, mode='fan_in', nonlinearity='linear')

        self.weight_path = sae_name
        
    def forward(self, x, return_latent=False):
        latent = F.relu(self.enc(x))
        x_hat = self.dec(latent)
        return (x_hat, latent) if return_latent else x_hat
    
    