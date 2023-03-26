"""
this extremely minimal Decision Transformer model is based on
the following causal transformer (GPT) implementation:

Misha Laskin's tweet:
https://twitter.com/MishaLaskin/status/1481767788775628801?cxt=HHwWgoCzmYD9pZApAAAA

and its corresponding notebook:
https://colab.research.google.com/drive/1NUBqyboDcGte5qAJKOl8gaJC28V_73Iv?usp=sharing

** the above colab notebook has a bug while applying masked_fill 
which is fixed in the following code
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal, Independent


class MaskedCausalAttention(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()

        self.n_heads = n_heads
        self.max_T = max_T

        self.q_net = nn.Linear(h_dim, h_dim)
        self.k_net = nn.Linear(h_dim, h_dim)
        self.v_net = nn.Linear(h_dim, h_dim)

        self.proj_net = nn.Linear(h_dim, h_dim)

        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)

        ones = torch.ones((max_T, max_T))
        mask = torch.tril(ones).view(1, 1, max_T, max_T)

        # register buffer makes sure mask does not get updated
        # during backpropagation
        self.register_buffer('mask',mask)

    def forward(self, x):
        B, T, C = x.shape # batch size, seq length, h_dim * n_heads

        N, D = self.n_heads, C // self.n_heads # N = num heads, D = attention dim

        # rearrange q, k, v as (B, N, T, D)
        q = self.q_net(x).view(B, T, N, D).transpose(1,2)
        k = self.k_net(x).view(B, T, N, D).transpose(1,2)
        v = self.v_net(x).view(B, T, N, D).transpose(1,2)

        # weights (B, N, T, T)
        weights = q @ k.transpose(2,3) / math.sqrt(D)
        # causal mask applied to weights
        weights = weights.masked_fill(self.mask[...,:T,:T] == 0, float('-inf'))
        # normalize weights, all -inf -> 0 after softmax
        normalized_weights = F.softmax(weights, dim=-1)

        # attention (B, N, T, D)
        attention = self.att_drop(normalized_weights @ v)

        # gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(1, 2).contiguous().view(B,T,N*D)

        out = self.proj_drop(self.proj_net(attention))
        return out


class Block(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()
        self.attention = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p)
        self.mlp = nn.Sequential(
                nn.Linear(h_dim, 4*h_dim),
                nn.GELU(),
                nn.Linear(4*h_dim, h_dim),
                nn.Dropout(drop_p),
            )
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, x):
        # Attention -> LayerNorm -> MLP -> LayerNorm
        x = x + self.attention(x) # residual
        x = self.ln1(x)
        x = x + self.mlp(x) # residual
        x = self.ln2(x)
        return x


class BCAgent(nn.Module):
    def __init__(self, state_dim, act_dim, n_blocks, h_dim, context_len,
                 n_heads, drop_p, max_timestep=4096):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.h_dim = h_dim

        ### transformer blocks
        input_seq_len = 2 * context_len
        blocks = [Block(h_dim, input_seq_len, n_heads, drop_p) for _ in range(n_blocks)]
        self.transformer = nn.Sequential(*blocks)

        ### projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_rtg = torch.nn.Linear(1, h_dim)
        self.embed_state = torch.nn.Linear(state_dim, h_dim)

        # # discrete actions
        # self.embed_action = torch.nn.Embedding(act_dim, h_dim)
        # use_action_tanh = False # False for discrete actions

        # continuous actions
        self.embed_action = torch.nn.Linear(act_dim, h_dim)
        use_action_tanh = True # True for continuous actions

        ### prediction heads
        self.predict_rtg = torch.nn.Linear(h_dim, 1)
        self.predict_state = torch.nn.Linear(h_dim, state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(h_dim, act_dim)] + ([nn.Tanh()] if use_action_tanh else []))
        )


    def forward(self, timesteps, states, actions):

        B, T, _ = states.shape

        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings

        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings

        # stack rtg, states and actions and reshape sequence as
        # (s_0, a_0,  s_1, a_1, s_2, a_2 ...)
        h = torch.stack(
            (state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(B, 2 * T, self.h_dim)

        h = self.embed_ln(h)

        # transformer and prediction
        h = self.transformer(h)

        # get h reshaped such that its size = (B x 2 x T x h_dim) ands
        # h[:, 0, t] is conditioned on the input sequence s_0, a_0 ... s_t
        # h[:, 1, t] is conditioned on the input sequence s_0, a_0 ... s_t, a_t
        # that is, for each timestep (t) we have 2 output embeddings from the transformer,
        # each conditioned on all previous timesteps plus 
        # the 2 input variables at that timestep (s_t, a_t) in sequence.
        h = h.reshape(B, T, 2, self.h_dim).permute(0, 2, 1, 3)

        # get predictions
        state_preds = self.predict_state(h[:,1])    # predict next state given s, a
        action_preds = self.predict_action(h[:,0])  # predict action given s

        return state_preds, action_preds

class GuideVAE(nn.Module):
    """Implementation of GuideVAE"""
    def __init__(self, class_size, feature_size, latent_size):
        super(GuideVAE, self).__init__()

        self.fc1x = nn.Linear(feature_size+class_size, 600)

        self.fc2_mu = nn.Linear(600, latent_size)
        self.fc2_log_std = nn.Linear(600, latent_size)

        self.fc2_mu_ = nn.Linear(300, latent_size)
        self.fc2_log_std_ = nn.Linear(300, latent_size)

        self.fc3 = nn.Linear(latent_size , 600)
        self.fc3_ = nn.Linear(latent_size , 300)
        self.fc4_mu = nn.Linear(600, feature_size + class_size)
        self.fc4_log_std = nn.Linear(600, feature_size + class_size)

    def encode(self, x, flag = True):
        if flag:
            h1 = self.fc1x(x)  # concat features and labels [64,240]->[64,300]
            z_mu = self.fc2_mu(h1)#[64,600]->[64,32]
            z_log_std = self.fc2_log_std(h1)#[64,600]->[64,32]
        else:
            h1 = self.fc1x(x)  # concat features and labels [64,240]->[64,300]
            z_mu = self.fc2_mu_(h1)#[64,300]->[64,32]
            z_log_std = self.fc2_log_std_(h1)#[64,300]->[64,32]
        return z_mu, z_log_std

    def decode(self, z):
        h3 = self.fc3(z)  # concat latents and labels [64,240]+[64,32]->[64,272]->[64,600]
        recon_mu = self.fc4_mu(h3)  # use sigmoid because the input image's pixel is between 0-1   [64,600]->[64,240]
        recon_log_std = self.fc4_log_std(h3)            #[64,600]->[64,240]
        return recon_mu, recon_log_std

    def reparametrize(self, z_mu, z_log_std):#重参数化
        z_std = torch.exp(z_log_std)
        z_eps = torch.randn_like(z_std)  # simple from standard normal distribution
        z = z_mu + z_eps * z_std
        return z

    def forward(self, x):
        z1_mu, z1_log_std = self.encode(x, flag = True)
        z1 = self.reparametrize(z1_mu, z1_log_std)
        recon_mu, recon_log_std = self.decode(z1)
        recon_log_std = torch.clamp(recon_log_std, -20, 2)
        recon_std = torch.exp(recon_log_std) 
        return recon_mu, recon_std, z1_mu, z1_log_std
    
    def reconstruct(self, x):
        z2_mu, z2_log_std = self.encode(x, 0, flag = False)
        z2 = self.reparametrize(z2_mu, z2_log_std)
        recon_mu2, recon_log_std2 = self.decode(z2, x)
        # recon_mu2 = torch.clamp(recon_mu2, -2, 1000)
        # recon_log_std2 = torch.clamp(recon_log_std2, -20, 2)
        return z2_mu, z2_log_std, recon_mu2, recon_log_std2

    def loss_function(self, recon_mu, recon_std, y) -> torch.Tensor:
        recon_distribution = Normal(recon_mu, recon_std)     
        #从分布中采样出概率密度，越大越好。
        recon_log_probs = recon_distribution.log_prob(y)
        recon_loss = -torch.mean(recon_log_probs) 
        # use "mean" may have a bad effect on gradients 。VAE是既要使得E[log p(x|z)]尽可能地大，也要使得KL散度尽可能的小，这两个是
        #相互独立的，但是他们都对训练起到很强的积极作用
        # kl_loss = torch.sum(z2_log_std-z1_log_std-0.5+(torch.exp(2*z1_log_std)+(z1_mu-z2_mu).pow(2))/(2*torch.exp(2*z2_log_std)))
        kl_loss = -0.5 * (1 + 2*torch.log(recon_std) - recon_mu.pow(2) - torch.exp(2*torch.log(recon_std)))
        kl_loss = torch.sum(kl_loss)
        loss = recon_loss #+ kl_loss
        return loss
class Starloss(nn.Module):
    def __init__(self) -> None:
        super().__init__()


    def log_normal(self, z, z2_mu, z2_std):
        z2_distribution = Normal(z2_mu, z2_std)
        loss_prob = -torch.mean(z2_distribution.log_prob(z))
        return loss_prob
    

    def forward(self, rewards, z, z1, ratio_a, ratio_b):
        loss =  -ratio_a*rewards+ratio_b*torch.nn.functional.mse_loss(z, z1) #+ self.log_normal(z, z2_mu, z2_std)+rewards*10
        return loss