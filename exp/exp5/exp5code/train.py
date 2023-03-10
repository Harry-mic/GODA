import argparse
import os
import sys
import random
import csv
from datetime import datetime
from utils import Dataset,TESTDataset
from model import GuideVAE,Starloss
import pdb

import numpy as np
import gym

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from decision_transformer.utils import D4RLTrajectoryDataset, evaluate_on_env, get_d4rl_normalized_score
from decision_transformer.model import DecisionTransformer

import sys

sys.path.append("..")

def goback(rtg, r, r_, r_mean, r_std, rtg_scale):
    r = r*r_std+r_mean
    r_ = r_*r_std+r_mean
    rtg = rtg*rtg_scale
    rtg_ = np.zeros_like(rtg)
    rtg = rtg.to(torch.device('cuda:0'))
    rtg_ = torch.from_numpy(rtg_).to(torch.device('cuda:0'))
    for i in range(20, 10, -1):
        temp = torch.zeros(64).to(torch.device('cuda:0'))
        for j in range(i - 11, 10):
            temp += r_[:,j] - r[:,j]
        rtg_[:,i-1] = rtg[:,i-1] + temp
    return rtg_[:,10:20]/rtg_scale

def train(args):
    seed=args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    dataset = args.dataset          # medium / medium-replay / medium-expert
    rtg_scale = args.rtg_scale      # normalize returns to go

    # use v3 env for evaluation because
    # Decision Transformer paper evaluates results on v3 envs

    if args.env == 'walker2d':
        env_name = 'Walker2d-v3'
        rtg_target = 5000
        env_d4rl_name = f'walker2d-{dataset}-v2'

    elif args.env == 'halfcheetah':
        env_name = 'HalfCheetah-v3'
        rtg_target = 6000
        env_d4rl_name = f'halfcheetah-{dataset}-v2'

    elif args.env == 'hopper':
        env_name = 'Hopper-v3'
        rtg_target = 3600
        env_d4rl_name = f'hopper-{dataset}-v2'

    else:
        raise NotImplementedError

    max_eval_ep_len = args.max_eval_ep_len  # max len of one episode
    num_eval_ep = args.num_eval_ep          # num of evaluation episodes

    batch_size = args.batch_size            # training batch size
    lr = args.lr                            # learning rate
    wt_decay = args.wt_decay                # weight decay
    warmup_steps = args.warmup_steps        # warmup steps for lr scheduler

    # total updates = max_train_iters x num_updates_per_iter
    max_train_iters = args.max_train_iters
    num_updates_per_iter = args.num_updates_per_iter

    context_len = args.context_len      # K in decision transformer
    n_blocks = args.n_blocks            # num of transformer blocks
    embed_dim = args.embed_dim          # embedding (hidden) dim of transformer
    n_heads = args.n_heads              # num of transformer heads
    dropout_p = args.dropout_p          # dropout probability
    star_batch_size = args.star_batch_size 

    # load data from this file
    dataset_path = f'{args.dataset_dir}/{env_d4rl_name}.pkl'

    # saves model and csv in this directory
    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # training and evaluation device
    device = torch.device(args.device)
    traj_dataset = Dataset(dataset_path, args.context_len,False)
    traj_test_dataset = TESTDataset(dataset_path, args.context_len,False)
    traj_star_dataset = D4RLTrajectoryDataset(dataset_path, context_len, rtg_scale)
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
    traj_star_loader = DataLoader(
                            traj_star_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True
                        )

    data_iter = iter(traj_data_loader)
    data_test_iter = iter(traj_data_test_loader)
    star_iter = iter(traj_star_loader)

    state_mean, state_std, reward_mean, reward_std, action_mean, action_std = traj_star_dataset.get_state_stats()

    start_time = datetime.now().replace(microsecond=0)
    start_time_str = start_time.strftime("%y-%m-%d-%H-%M-%S")

    prefix = "dt_" + env_d4rl_name

    save_model_name =  prefix + "_model_" + start_time_str + ".pt"
    save_model_path = os.path.join(log_dir, save_model_name)
    save_best_model_path = save_model_path[:-3] + "_best.pt"

    log_csv_name = prefix + "_log_" + start_time_str + ".csv"
    log_csv_path = os.path.join(log_dir, log_csv_name)

    csv_writer = csv.writer(open(log_csv_path, 'a', 1))
    csv_header = (["duration", "num_updates", "action_loss",
                   "eval_avg_reward", "eval_avg_ep_len", "eval_d4rl_score"])

    csv_writer.writerow(csv_header)

    print("=" * 60)
    print("start time: " + start_time_str)
    print("=" * 60)

    print("device set to: " + str(device))
    print("dataset path: " + dataset_path)
    print("model save path: " + save_model_path)
    print("log csv save path: " + log_csv_path)


    # ## get state stats from dataset
    # state_mean, state_std = traj_dataset.get_state_stats()

    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model_guide = GuideVAE(args.feature_size, args.class_size, args.latent_size).to(device)
    optimizer_guide = torch.optim.AdamW(
                        model_guide.parameters(),
                        lr=lr,
                        weight_decay=wt_decay
                    )
    scheduler_guide = torch.optim.lr_scheduler.LambdaLR(
                            optimizer_guide,
                            lambda steps: min((steps+1)/warmup_steps, 1)
                    )

    model = DecisionTransformer(
                state_dim=state_dim,
                act_dim=act_dim,
                n_blocks=n_blocks,
                h_dim=embed_dim,
                context_len=context_len,
                n_heads=n_heads,
                drop_p=dropout_p,
            ).to(device)

    optimizer = torch.optim.AdamW(
                        model.parameters(),
                        lr=lr,
                        weight_decay=wt_decay
                    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
                            optimizer,
                            lambda steps: min((steps+1)/warmup_steps, 1)
                        )

    max_d4rl_score = -1.0
    total_updates = 0

#******************************************************************************************?????????????????????************************************************************************
#???????????????z*model???
    for i_iter in range(max_train_iters):
        log_GODA_losses = []
        if not args.eval:
            for _ in range(num_updates_per_iter):
                try:
                    states, actions, rewards = next(data_iter)
                except StopIteration:
                    data_iter = iter(traj_data_loader)
                    states, actions, rewards = next(data_iter)

                states = states.reshape(64,-1).to(device)          # B x T x state_dim     [64,340]
                actions = actions.reshape(64,-1).to(device)        # B x T x act_dim       [64,120]
                rewards = rewards.reshape(64,-1).to(device)        #                       [64,20]       
                
                #???????????????????????????????????????
                feature = torch.cat([states[:,0:int(state_dim*(context_len/2))],
                                     actions[:,0:int(act_dim*(context_len/2))],
                                     rewards[:,0:int((context_len/2))]],dim=1) #s1a1r1~s5a5r5     #[64,240]
                feature_class = torch.cat([states[:,int(state_dim*(context_len/2)):state_dim*context_len],
                                           actions[:,int(act_dim*(context_len/2)):act_dim*context_len],
                                           rewards[:,int((context_len/2)):context_len]],dim=1) #s6a6r6~s10a10r10      [64,240]                                     #

                recon_mu, recon_std, z1_mu, z1_log_std = model_guide.forward(feature, feature_class)
                z2_mu, z2_log_std, recon_mu2, recon_log_std2 = model_guide.reconstruct(feature)
                GODA_loss = model_guide.loss_function(recon_mu, recon_std, feature_class, z1_mu, z1_log_std, z2_mu, z2_log_std)

                optimizer_guide.zero_grad()
                GODA_loss.backward()

                torch.nn.utils.clip_grad_norm_(model_guide.parameters(), 0.25)
                optimizer_guide.step()
                scheduler_guide.step()

                log_GODA_losses.append(GODA_loss.detach().cpu().item())
        mean_GODA_loss = np.mean(log_GODA_losses)
        total_updates += num_updates_per_iter
        log_str = (
            "num of updates: " + str(total_updates) + '\n' +
            "GODA loss: " +  format(mean_GODA_loss, ".5f") + '\n'
        )
        print(log_str)


#Test the training result
    final_GODA_loss=[]
    for _ in range(num_updates_per_iter):
        try:
            states, actions, rewards = next(data_test_iter)
        except StopIteration:
            data_test_iter = iter(traj_data_test_loader)
            states, actions, rewards = next(data_test_iter)

        states = states.reshape(64,-1).to(device)          # B x T x state_dim     [64,170]
        actions = actions.reshape(64,-1).to(device)        # B x T x act_dim       [64,60]
        rewards = rewards.reshape(64,-1).to(device)  

        feature = torch.cat([states[:,0:int(state_dim*(context_len/2))],
                                    actions[:,0:int(act_dim*(context_len/2))],
                                    rewards[:,0:int((context_len/2))]],dim=1) #s1a1r1~s5a5r5     
                                                                        #
        feature_class = torch.cat([states[:,int(state_dim*(context_len/2)):state_dim*context_len],
                                    actions[:,int(act_dim*(context_len/2)):act_dim*context_len],
                                    rewards[:,int((context_len/2)):context_len]],dim=1) #s6a6r6~s10a10r10                                                 #
            
        recon_mu, recon_std, z1_mu, z1_log_std = model_guide.forward(feature, feature_class)
        z2_mu, z2_log_std, recon_mu2, recon_log_std2 = model_guide.reconstruct(feature)
        GODA_loss = model_guide.loss_function(recon_mu, recon_std, feature_class, z1_mu, z1_log_std, z2_mu, z2_log_std)

        final_GODA_loss.append(GODA_loss.detach().cpu().item())
    mean_GODA_final_loss = np.mean(final_GODA_loss)
    log_str = (
            "GODA test loss: " +  format(mean_GODA_final_loss, ".5f") + '\n'
        )
    print(log_str)
#******************************************************************************************?????????????????????************************************************************************

    for i_train_iter in range(max_train_iters):

        log_action_losses = []
        model.train()

        for _ in range(num_updates_per_iter):
            try:
                timesteps, states, actions, returns_to_go, traj_mask, rewards = next(star_iter)
            except StopIteration:
                star_iter = iter(traj_star_loader)
                timesteps, states, actions, returns_to_go, traj_mask, rewards = next(star_iter)

            #???????????????????????????????????????********************************************************************************************************************************
            states = states.reshape(star_batch_size ,-1).to(device)          # B x T x state_dim     [64,170]                                                       # 
            actions = actions.reshape(star_batch_size ,-1).to(device)        # B x T x act_dim       [64,60]                                                        #
            rewards = rewards.reshape(star_batch_size ,-1).to(device)        #                       [64,10]                                                        #
                                                                                                                                                                    
            feature = torch.cat([states[:,0:int(state_dim*(context_len/2))],  
                                    actions[:,0:int(act_dim*(context_len/2))],                                                                                      #
                                        rewards[:,0:int((context_len/2))]],dim=1) #s1a1r1~s5a5r5                                                                       #    

            feature_class = torch.cat([states[:,int(state_dim*(context_len/2)):state_dim*context_len],                                                              #
                                        actions[:,int(act_dim*(context_len/2)):act_dim*context_len],                                                                #
                                        rewards[:,int((context_len/2)):context_len]],dim=1) #s6a6r6~s10a10r10                                                       #
                                                                                                                                                                    
            z1_mu, z1_log_std = model_guide.encode(feature, feature_class, True)                                                                                    #
            z2_mu, z2_log_std = model_guide.encode(feature, 0, False)                                                                                               #
            z2_log_std = torch.clamp(z2_log_std, -20, 2)                                                                                                            #
            z2_std = torch.exp(z2_log_std)                                                                                                                          #

            # recon_mu_ori,recon_log_std_ori = model_guide.decode(z1_mu,feature)                                                                                    #
            z = z1_mu.clone().detach()                                                                                                                              #
            z.requires_grad = True                                                                                                                                  #
            scene_optim = torch.optim.Adam([z], lr=lr)                                                                                                              #
            loss_star_function = Starloss()                                                                                                                         #
            loss =[]                                                                                                                                                #
            
            #?????????64???????????????????????????z                                                                                                                              #                
            for i in range(200):                                                                                                                                    #
                recon_mu,recon_log_std = model_guide.decode(z,feature)                                                                                              #
                rewards_ = torch.mean(recon_mu[:,(state_dim+act_dim)*int(context_len/2):])#rewards: s6'~s10'                                                        #
                loss_star = loss_star_function.forward(rewards_, z, z1_mu, z2_mu, z2_std)                                                                           #
                loss.append(loss_star.detach().cpu().item())                                                                                                        #
                scene_optim.zero_grad()                                                                                                                             #                 
                loss_star.backward(retain_graph=True)                                                                                                               #
                scene_optim.step()                                                                                                                                  #
            
        #????????????                                                                                                                                                    #
            #??????s6'a6'r6'~s10'a10'r10'                                                                                                                             #
            timesteps = timesteps.to(device)    # B x T                                                                                                             #
            states_ = recon_mu[:,0:int(state_dim*(context_len/2))]                                                                                               #
            actions_ = recon_mu[:,state_dim*int(context_len/2):(state_dim+act_dim)*int(context_len/2)].cpu().detach().numpy()
            actions_ =  (actions_.reshape(64,10,6)*action_std+action_mean).reshape(64,-1)
            rewards_ = recon_mu[:,(state_dim+act_dim)*int(context_len/2):]
            returns_to_go_ = goback(    #[64,10]
                returns_to_go,   #[64,20]
                rewards[:,int((context_len/2)):context_len],   #[64,10]
                rewards_,   #[64,10]
                reward_mean, 
                reward_std,
                rtg_scale).to(device)  
            
            #??????1~5+6'~10'
            states = torch.cat([states[:,0:int(state_dim*(context_len/2))],
                                states_]).reshape(star_batch_size, context_len, state_dim).reshape(64,20,17).to(device)
            actions = torch.cat([actions[:,0:int(act_dim*(context_len/2))],
                                    torch.from_numpy(actions_).to(device)]).reshape(star_batch_size, context_len, act_dim).reshape(64,20,6).to(device) 
            
            returns_to_go = torch.cat([returns_to_go[:,0:int((context_len/2))].to(device),
                                        returns_to_go_]).reshape(star_batch_size , context_len, 1).to(device)  
                        
            traj_mask = traj_mask.to(device)    # B x T                                                                                                             #
            action_target = torch.clone(actions).detach().to(device)                                                                                                #
            #********************************************************************************************************************************************************
            #???star???????????????
            state_preds, action_preds, return_preds = model.forward(
                                                            timesteps=timesteps,
                                                            states=states,
                                                            actions=actions,
                                                            returns_to_go=returns_to_go
                                                        )
            # only consider non padded elements
            action_preds = action_preds.view(-1, act_dim)[traj_mask.view(-1,) > 0]
            action_target = action_target.view(-1, act_dim)[traj_mask.view(-1,) > 0]

            action_loss = F.mse_loss(action_preds, action_target, reduction='mean')

            optimizer.zero_grad()
            action_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            scheduler.step()

            log_action_losses.append(action_loss.detach().cpu().item())

        # evaluate action accuracy
        results = evaluate_on_env(model, device, context_len, env, rtg_target, rtg_scale,
                                num_eval_ep, max_eval_ep_len, state_mean, state_std)

        eval_avg_reward = results['eval/avg_reward']
        eval_avg_ep_len = results['eval/avg_ep_len']
        eval_d4rl_score = get_d4rl_normalized_score(results['eval/avg_reward'], env_name) * 100

        mean_action_loss = np.mean(log_action_losses)
        time_elapsed = str(datetime.now().replace(microsecond=0) - start_time)

        total_updates += num_updates_per_iter

        log_str = ("=" * 60 + '\n' +
                "time elapsed: " + time_elapsed  + '\n' +
                "num of updates: " + str(total_updates) + '\n' +
                "action loss: " +  format(mean_action_loss, ".5f") + '\n' +
                "eval avg reward: " + format(eval_avg_reward, ".5f") + '\n' +
                "eval avg ep len: " + format(eval_avg_ep_len, ".5f") + '\n' +
                "eval d4rl score: " + format(eval_d4rl_score, ".5f")
            )

        print(log_str)

        log_data = [time_elapsed, total_updates, mean_action_loss,
                    eval_avg_reward, eval_avg_ep_len,
                    eval_d4rl_score]

        csv_writer.writerow(log_data)

        # save model
        print("max d4rl score: " + format(max_d4rl_score, ".5f"))
        if eval_d4rl_score >= max_d4rl_score:
            print("saving max d4rl score model at: " + save_best_model_path)
            torch.save(model.state_dict(), save_best_model_path)
            max_d4rl_score = eval_d4rl_score

        print("saving current model at: " + save_model_path)
        torch.save(model.state_dict(), save_model_path)


    print("=" * 60)
    print("finished training!")
    print("=" * 60)
    end_time = datetime.now().replace(microsecond=0) 
    time_elapsed = str(end_time - start_time)
    end_time_str = end_time.strftime("%y-%m-%d-%H-%M-%S")
    print("started training at: " + start_time_str)
    print("finished training at: " + end_time_str)
    print("total training time: " + time_elapsed)
    print("max d4rl score: " + format(max_d4rl_score, ".5f"))
    print("saved max d4rl score model at: " + save_best_model_path)
    print("saved last updated model at: " + save_model_path)
    print("=" * 60)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default='halfcheetah')
    parser.add_argument('--dataset', type=str, default='medium')
    parser.add_argument('--rtg_scale', type=int, default=1000)

    parser.add_argument('--max_eval_ep_len', type=int, default=1000)
    parser.add_argument('--num_eval_ep', type=int, default=10)

    parser.add_argument('--dataset_dir', type=str, default='data/')
    parser.add_argument('--log_dir', type=str, default='dt_runs/')

    parser.add_argument('--context_len', type=int, default=20)
    parser.add_argument('--n_blocks', type=int, default=3)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=1)
    parser.add_argument('--dropout_p', type=float, default=0.1)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wt_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)

    parser.add_argument('--max_train_iters', type=int, default=200)
    parser.add_argument('--num_updates_per_iter', type=int, default=100)

    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--seed',type=int, default=0)
    parser.add_argument('--log_fn',type=str,default='default')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--load_model_path', type=str,default='')
    parser.add_argument('--squences_length', type=int, default=10)
    parser.add_argument('--recovery_length', type=int, default=5)
    parser.add_argument('--total_episodes', type=int, default=2186)
    parser.add_argument('--star_batch_size', type=int, default=64)
    # parser.add_argument('--log_fn',type=str,default='default')
    parser.add_argument('--feature_size', type=int, default=240)
    parser.add_argument('--class_size', type=int, default=240)
    parser.add_argument('--latent_size', type=int, default=64)
    parser.add_argument('--batch_nums', type=int, default=1000)
    parser.add_argument('--traj_length', type=int, default=1000)

    args = parser.parse_args()

    train(args)
