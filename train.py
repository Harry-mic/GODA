import argparse
import os
import random
import csv
from datetime import datetime
import numpy as np
import gym
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import GuideVAE, PredictVAE
from utils import Dataset,TESTDataset
import pdb

def train(args):
#finish
    seed=args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

#finish
    env_d4rl_name = args.dataset
    batch_size = args.batch_size            # training batch size
    lr = args.lr                            # learning rate
    max_train_iters = args.max_train_iters
    num_updates_per_iter = args.num_updates_per_iter
    dataset_path = f'{args.dataset_dir}/{env_d4rl_name}.pkl'
    wt_decay = args.wt_decay                # weight decay
    warmup_steps = args.warmup_steps        # warmup steps for lr scheduler

#finish
    # saves model and csv in this directory
    log_dir = args.log_dir
    model_log_dir=log_dir+'/models'
    csv_log_dir=log_dir+'/logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(model_log_dir):
        os.makedirs(model_log_dir)
    if not os.path.exists(csv_log_dir):
        os.makedirs(csv_log_dir)

#finish
    device = torch.device(args.device)
    traj_dataset = Dataset(dataset_path, args.context_len,False)
    traj_test_dataset = TESTDataset(dataset_path, args.context_len,False)
    traj_data_loader = DataLoader(
                            traj_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True
                        )
    traj_data_test_loader = DataLoader(
                            traj_test_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True
                        )
    data_iter = iter(traj_data_loader)
    data_test_iter = iter(traj_data_test_loader)

#finish
    start_time = datetime.now().replace(microsecond=0)
    start_time_str = start_time.strftime("%y-%m-%d-%H-%M-%S")

#finish
    model_guide = GuideVAE(args.feature_size, args.class_size, args.latent_size).to(device)
    model_predict =  PredictVAE(args.feature_size, args.latent_size).to(device)
    optimizer_guide = torch.optim.AdamW(
                        model_guide.parameters(),
                        lr=lr,
                        weight_decay=wt_decay
                    )
    scheduler_guide = torch.optim.lr_scheduler.LambdaLR(
                            optimizer_guide,
                            lambda steps: min((steps+1)/warmup_steps, 1)
                        )
    optimizer_predict = torch.optim.AdamW(
                        model_predict.parameters(),
                        lr=lr,
                        weight_decay=wt_decay
                    )
    scheduler_predict = torch.optim.lr_scheduler.LambdaLR(
                            optimizer_predict,
                            lambda steps: min((steps+1)/warmup_steps, 1)
                        )
    total_updates = 0


    for i_iter in range(max_train_iters):
        log_GODA_losses = []
        if not args.eval:
            for _ in range(num_updates_per_iter):
                try:
                    states, actions, rewards = next(data_iter)
                except StopIteration:
                    data_iter = iter(traj_data_loader)
                    states, actions, rewards = next(data_iter)

                states = states.reshape(64,-1).to(device)          # B x T x state_dim     [64,170]
                actions = actions.reshape(64,-1).to(device)        # B x T x act_dim       [64,60]
                rewards = rewards.reshape(64,-1).to(device)        #                       [64,10]       
#此处需要将三个变量合并起来
                feature = torch.cat([states[:,0:85],actions[:,0:30],rewards[:,0:5]],dim=1) #s1a1r1~s5a5r5
                feature_class = torch.cat([states[:,85:170],actions[:,30:60],rewards[:,5:10]],dim=1) #s6a6r6~s10a10r10

                # states_target = torch.clone(states).detach().to(device)
                # actions_target = torch.clone(actions).detach().to(device)
                # rewards_target = torch.clone(rewards).detach().to(device)

                recon_mu, recon_std, z1_mu, z1_log_std = model_guide.forward(feature, feature_class)
                GODA_loss1 = model_guide.loss_function(recon_mu, recon_std, feature_class, z1_mu, z1_log_std)

                z2_mu, z2_log_std = model_predict.forward(feature)
                GODA_loss2 = model_predict.loss_function(z1_mu, z1_log_std, z2_mu, z2_log_std)
                
                GODA_loss = torch.mean(GODA_loss1+GODA_loss2)

                optimizer_guide.zero_grad()
                optimizer_predict.zero_grad()
                GODA_loss.backward()
                torch.nn.utils.clip_grad_norm_(model_guide.parameters(), 0.25)
                torch.nn.utils.clip_grad_norm_(model_predict.parameters(), 0.25)
                optimizer_guide.step()
                scheduler_guide.step()
                optimizer_predict.step()
                scheduler_predict.step()

                log_GODA_losses.append(GODA_loss.detach().cpu().item())
        mean_GODA_loss = np.mean(log_GODA_losses)
        total_updates += num_updates_per_iter
        log_str = (
            "num of updates: " + str(total_updates) + '\n' +
            "GODA loss: " +  format(mean_GODA_loss, ".5f") + '\n'
        )
        print(log_str)

#Test the training result
    # final_GODA_loss=[]
    # for _ in range(num_updates_per_iter):
    #     try:
    #         states, actions, rewards = next(data_test_iter)
    #     except StopIteration:
    #         data_test_iter = iter(traj_data_test_loader)
    #         states, actions, rewards = next(data_test_iter)

    #     states = states.to(device)          # B x T x state_dim
    #     actions = actions.to(device)        # B x T x act_dim
    #     rewards = rewards.to(device)


    #     states_target = torch.clone(states).detach().to(device)
    #     actions_target = torch.clone(actions).detach().to(device)
    #     rewards_target = torch.clone(rewards).detach().to(device)

    #     new_states_log_probs, new_actions_log_probs, new_rewards_log_probs = \
    #     model.forward(states, actions, rewards, states_target, actions_target, rewards_target)

    #     GODA_loss = -torch.mean(new_states_log_probs) - torch.mean(new_actions_log_probs) - torch.mean(new_rewards_log_probs)
    #     final_GODA_loss.append(GODA_loss.detach().cpu().item())
    # mean_GODA_final_loss = np.mean(final_GODA_loss)
    # log_str = (
    #         "GODA test loss: " +  format(mean_GODA_final_loss, ".5f") + '\n'
    #     )
    # print(log_str)
    # new_states_mean, new_actions_mean, new_rewards_mean = \
    #             model.get_value(states, actions, rewards, states_target, actions_target, rewards_target)
    # new_states_mean = pd.DataFrame(new_states_mean.detach().cpu())
    # new_actions_mean = pd.DataFrame(new_actions_mean.detach().cpu())
    # new_rewards_mean = pd.DataFrame(new_rewards_mean.detach().cpu())
    # new_states_mean.to_csv("~/code/GODA/yuce/new_states_mean.csv")
    # new_actions_mean.to_csv("~/code/GODA/yuce/new_actions_mean.csv")
    # new_rewards_mean.to_csv("~/code/GODA/yuce/new_rewards_mean.csv")
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default='HalfCheetah-v3')
    parser.add_argument('--seed',type=int, default=0)
    parser.add_argument('--dataset_dir', type=str, default='data/')
    parser.add_argument('--dataset', type=str, default='halfcheetah_medium-v2')
    parser.add_argument('--log_dir', type=str, default='logs/sdt_runs/')
    parser.add_argument('--log_fn',type=str,default='default')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--load_model_path', type=str,default='')
    parser.add_argument('--squences_length', type=int, default=10)
    parser.add_argument('--recovery_length', type=int, default=5)
    parser.add_argument('--context_len', type=int, default=10)
    parser.add_argument('--total_episodes', type=int, default=2186)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_train_iters', type=int, default=30)
    parser.add_argument('--num_updates_per_iter', type=int, default=2000)
    parser.add_argument('--device', type=str, default='cuda')
    # parser.add_argument('--log_fn',type=str,default='default')
    parser.add_argument('--wt_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--feature_size', type=int, default=120)
    parser.add_argument('--class_size', type=int, default=120)
    parser.add_argument('--latent_size', type=int, default=32)

    args = parser.parse_args()
    train(args)