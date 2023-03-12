import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform
import pdb
import numpy as np

class GODArecovery(nn.Module):
    def __init__(self, squences_length, recovery_length):
        super().__init__()

        self.squences_length = squences_length
        self.recovery_length = recovery_length
        self.log_std_min = -20
        self.log_std_max = 2

#恢复states，输入为[64,17,10]输出为[64,17,5]
        self.recovery_states_mean = nn.Sequential(
            nn.Linear(self.squences_length, self.recovery_length),
        )

        self.recovery_states_logstd = nn.Sequential(
            nn.Linear(self.squences_length, self.recovery_length),
        )
#恢复actions，输入为[64,6,10]输出为[64,6,5]        
        self.recovery_actions_mean = nn.Sequential(
            nn.Linear(self.squences_length, self.recovery_length),
        )

        self.recovery_actions_logstd = nn.Sequential(
            nn.Linear(self.squences_length, self.recovery_length),
        )
#恢复actions，输入为[64,1,10]输出为[64,1,5]  
        self.recovery_rewards_mean = nn.Sequential(
            nn.Linear(self.squences_length, self.recovery_length),
        )
        self.recovery_rewards_logstd = nn.Sequential(
            nn.Linear(self.squences_length, self.recovery_length),
        )


    def forward(self, states, actions, rewards, target_states, target_actions, target_rewards):
        rewards = rewards.reshape(64,10,1)
        target_rewards = target_rewards.reshape(64,10,1)
#预测新序列的states、actions、rewards的均值
        new_states_mean = self.recovery_states_mean(states.permute(0,2,1)).permute(0,2,1)#[64,5,17]
        new_actions_mean = self.recovery_actions_mean(actions.permute(0,2,1)).permute(0,2,1)#[64,5,6]
        new_rewards_mean = self.recovery_rewards_mean(rewards.permute(0,2,1)).permute(0,2,1)#[64,5,1]
#预测新序列的states、actions、rewards的标准差
        new_states_logstd = self.recovery_states_logstd(states.permute(0,2,1)).permute(0,2,1)#[64,5,17]
        new_states_logstd = torch.clamp(new_states_logstd, self.log_std_min, self.log_std_max)
        new_states_stds = torch.exp(new_states_logstd)

        new_actions_logstd = self.recovery_actions_logstd(actions.permute(0,2,1)).permute(0,2,1)#[64,5,6]
        new_actions_logstd = torch.clamp(new_actions_logstd, self.log_std_min, self.log_std_max)
        new_actions_stds = torch.exp(new_actions_logstd)

        new_rewards_logstd = self.recovery_rewards_logstd(rewards.permute(0,2,1)).permute(0,2,1)#[64,5,1]
        new_rewards_logstd = torch.clamp(new_rewards_logstd, self.log_std_min, self.log_std_max)
        new_rewards_stds = torch.exp(new_rewards_logstd)
#化为分布形式
        new_states_distribution = Normal(new_states_mean, new_states_stds)#Normal 表示构造一个正态分布
        new_actions_distribution = Independent(TransformedDistribution(Normal(new_actions_mean, new_actions_stds), TanhTransform(cache_size=1)),1)#Normal 表示构造一个正态分布
        new_rewards_distribution = Normal(new_rewards_mean,new_rewards_stds)
#查看分布预估是否准确
#将原动作输入新生成的分布中，表示从新生成的分布中采样出原动作的概率的对数。
#如果从新分布中采样出原动作的概率为1，说明这个新分布生成的非常好，此时的对数概率密度为0，也就是loss为0.如果小于1，就说明新分布生成的不够好，需要改进，因此就会有loss       
        new_states_log_probs = new_states_distribution.log_prob(target_states[:,5:10])#这个地方报错，因为我的target_states没有归一化

        eps = torch.finfo(target_actions.dtype).eps
        target_actions = torch.clamp(target_actions, -1+eps, 1-eps)
        new_actions_log_probs = new_actions_distribution.log_prob(target_actions[:,5:10])

        new_rewards_log_probs = new_rewards_distribution.log_prob(target_rewards[:,5:10])

        return new_states_log_probs, new_actions_log_probs, new_rewards_log_probs
    
    def get_value(self, states, actions, rewards, target_states, target_actions, target_rewards):
        rewards = rewards.reshape(64,10,1)
        target_rewards = target_rewards.reshape(64,10,1)
#预测新序列的states、actions、rewards的均值
        new_states_mean = self.recovery_states_mean(states.permute(0,2,1)).permute(0,2,1)#[64,5,17]
        new_actions_mean = self.recovery_actions_mean(actions.permute(0,2,1)).permute(0,2,1)#[64,5,6]
        new_rewards_mean = self.recovery_rewards_mean(rewards.permute(0,2,1)).permute(0,2,1)#[64,5,1]

        return  new_states_mean, new_actions_mean, new_rewards_mean


class GuideVAE(nn.Module):
    """Implementation of GuideVAE"""
    def __init__(self, feature_size, class_size, latent_size):
        super(GuideVAE, self).__init__()

        self.fc1x = nn.Linear(feature_size, 150)
        self.fc1y = nn.Linear(class_size, 150)

        self.fc2_mu = nn.Linear(300, latent_size)
        self.fc2_log_std = nn.Linear(300, latent_size)

        self.fc2_mu_ = nn.Linear(150, latent_size)
        self.fc2_log_std_ = nn.Linear(150, latent_size)

        self.fc3 = nn.Linear(latent_size + class_size, 300)
        self.fc3_ = nn.Linear(latent_size , 300)
        self.fc4_mu = nn.Linear(300, feature_size)
        self.fc4_log_std = nn.Linear(300, feature_size)

    def encode(self, x, y, flag = True):
        if flag:
            h1x = self.fc1x(x)  # concat features and labels [64,120]->[64,150]
            h1y = self.fc1y(y)
            h1 = torch.cat([h1x,h1y], dim=1)
            z_mu = self.fc2_mu(h1)#[64,300]->[64,32]
            z_log_std = self.fc2_log_std(h1)#[64,300]->[64,32]
        else:
            h1 = self.fc1x(x)  # concat features and labels [64,120]->[64,200]
            z_mu = self.fc2_mu_(h1)#[64,200]->[64,32]
            z_log_std = self.fc2_log_std_(h1)#[64,200]->[64,32]
        return z_mu, z_log_std

    def decode(self, z, x):
        h3 = self.fc3(torch.cat([z, x], dim=1))  # concat latents and labels
        recon_mu = self.fc4_mu(h3)  # use sigmoid because the input image's pixel is between 0-1
        recon_log_std = self.fc4_log_std(h3)
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
        recon_log_std = torch.clamp(recon_log_std, -20, 2)
        recon_std = torch.exp(recon_log_std) 
        return recon_mu, recon_std, z1_mu, z1_log_std
    
    def reconstruct(self, x):
        z2_mu, z2_log_std = self.encode(x, 0, flag = False)
        z2 = self.reparametrize(z2_mu, z2_log_std)
        recon_mu2, recon_log_std2 = self.decode(z2, x)
        return z2_mu, z2_log_std, recon_mu2, recon_log_std2

    def loss_function(self, recon_mu, recon_std, y, z1_mu, z1_log_std, z2_mu, z2_log_std) -> torch.Tensor:
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
    
    


