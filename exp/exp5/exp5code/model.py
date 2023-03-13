import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform
import pdb
import numpy as np
class GuideVAE(nn.Module):
    """Implementation of GuideVAE"""
    def __init__(self, class_size, feature_size, latent_size):
        super(GuideVAE, self).__init__()

        self.fc1x = nn.Linear(feature_size, 300)
        self.fc1y = nn.Linear(class_size, 300)

        self.fc2_mu = nn.Linear(600, latent_size)
        self.fc2_log_std = nn.Linear(600, latent_size)

        self.fc2_mu_ = nn.Linear(300, latent_size)
        self.fc2_log_std_ = nn.Linear(300, latent_size)

        self.fc3 = nn.Linear(latent_size + class_size, 600)
        self.fc3_ = nn.Linear(latent_size , 300)
        self.fc4_mu = nn.Linear(600, feature_size)
        self.fc4_log_std = nn.Linear(600, feature_size)

    def encode(self, x, y, flag = True):
        if flag:
            h1x = self.fc1x(x)  # concat features and labels [64,240]->[64,300]
            h1y = self.fc1y(y)  # concat features and labels [64,240]->[64,300]
            h1 = torch.cat([h1x,h1y], dim=1)# [64,600]
            z_mu = self.fc2_mu(h1)#[64,600]->[64,32]
            z_log_std = self.fc2_log_std(h1)#[64,600]->[64,32]
        else:
            h1 = self.fc1x(x)  # concat features and labels [64,240]->[64,300]
            z_mu = self.fc2_mu_(h1)#[64,300]->[64,32]
            z_log_std = self.fc2_log_std_(h1)#[64,300]->[64,32]
        return z_mu, z_log_std

    def decode(self, z, x):
        h3 = self.fc3(torch.cat([z, x], dim=1))  # concat latents and labels [64,240]+[64,32]->[64,272]->[64,600]
        recon_mu = self.fc4_mu(h3)  # use sigmoid because the input image's pixel is between 0-1   [64,600]->[64,240]
        recon_log_std = self.fc4_log_std(h3)            #[64,600]->[64,240]
        return recon_mu, recon_log_std
        # else:
        #     h3 = self.fc3_(z) # concat latents and labels
        #     recon_mu = self.fc4_mu(h3) # use sigmoid because the input image's pixel is between 0-1
        #     recon_log_std = self.fc4_log_std(h3)

    def reparametrize(self, z_mu, z_log_std):#重参数化
        z_std = torch.exp(z_log_std)
        z_eps = torch.randn_like(z_std)  # simple from standard normal distribution
        z = z_mu + z_eps * z_std
        return z

    def forward(self, x, y):
        z1_mu, z1_log_std = self.encode(x, y, flag = True)
        z1 = self.reparametrize(z1_mu, z1_log_std)
        recon_mu, recon_log_std = self.decode(z1, x)
        recon_mu = torch.clamp(recon_mu, -2, 1000)
        recon_log_std = torch.clamp(recon_log_std, -20, 2)
        recon_std = torch.exp(recon_log_std) 
        return recon_mu, recon_std, z1_mu, z1_log_std
    
    def reconstruct(self, x):
        z2_mu, z2_log_std = self.encode(x, 0, flag = False)
        z2 = self.reparametrize(z2_mu, z2_log_std)
        recon_mu2, recon_log_std2 = self.decode(z2, x)
        recon_mu2 = torch.clamp(recon_mu2, -2, 1000)
        recon_log_std2 = torch.clamp(recon_log_std2, -20, 2)
        return z2_mu, z2_log_std, recon_mu2, recon_log_std2

    def loss_function(self, recon_mu, recon_std, y, z1_mu, z1_log_std, z2_mu, z2_log_std) -> torch.Tensor:
        a1 = np.isnan(recon_mu.cpu().detach().numpy()).any()
        a2 = np.isnan(recon_std.cpu().detach().numpy()).any()
        recon_distribution = Normal(recon_mu, recon_std)     
        #从分布中采样出概率密度，越大越好。
        recon_log_probs = recon_distribution.log_prob(y)
        recon_loss = -torch.mean(recon_log_probs) #mean(|-(lnp-ln0.5)|)
        # use "mean" may have a bad effect on gradients 。VAE是既要使得E[log p(x|z)]尽可能地大，也要使得KL散度尽可能的小，这两个是
        #相互独立的，但是他们都对训练起到很强的积极作用
        kl_loss = torch.sum(z2_log_std-z1_log_std-0.5+(torch.exp(2*z1_log_std)+(z1_mu-z2_mu).pow(2))/(2*torch.exp(2*z2_log_std)))
        loss = recon_loss + kl_loss
        return loss
    
class Starloss(nn.Module):
    def __init__(self) -> None:
        super().__init__()


    def log_normal(self, z, z2_mu, z2_std):
        z2_distribution = Normal(z2_mu, z2_std)
        loss_prob = -torch.mean(z2_distribution.log_prob(z))
        return loss_prob
    

    def forward(self, rewards, z, z1, z2_mu, z2_std):
        loss = -rewards + 1000*torch.nn.functional.mse_loss(z, z1) #+ 10*self.log_normal(z, z2_mu, z2_std)
        return loss
    
    


