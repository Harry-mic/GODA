import random
import time
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from imitation_learning.d4rl_infos import REF_MIN_SCORE, REF_MAX_SCORE, D4RL_DATASET_STATS


def discount_cumsum(x, gamma):
    disc_cumsum = np.zeros_like(x)
    disc_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        disc_cumsum[t] = x[t] + gamma * disc_cumsum[t+1]
    return disc_cumsum


def get_d4rl_normalized_score(score, env_name):
    env_key = env_name.split('-')[0].lower()
    assert env_key in REF_MAX_SCORE, f'no reference score for {env_key} env to calculate d4rl score'
    return (score - REF_MIN_SCORE[env_key]) / (REF_MAX_SCORE[env_key] - REF_MIN_SCORE[env_key])


def get_d4rl_dataset_stats(env_d4rl_name):
    return D4RL_DATASET_STATS[env_d4rl_name]


def evaluate_on_env(model, device, context_len, env,
                    num_eval_ep=10, max_test_ep_len=1000,
                    state_mean=None, state_std=None, render=False):

    eval_batch_size = 1  # required for forward pass

    results = {}
    total_reward = 0
    total_cost = 0
    total_timesteps = 0

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    if state_mean is None:
        state_mean = torch.zeros((state_dim,)).to(device)
    else:
        state_mean = torch.from_numpy(state_mean).to(device)

    if state_std is None:
        state_std = torch.ones((state_dim,)).to(device)
    else:
        state_std = torch.from_numpy(state_std).to(device)

    # same as timesteps used for training the transformer
    # also, crashes if device is passed to arange()
    timesteps = torch.arange(start=0, end=max_test_ep_len, step=1)
    timesteps = timesteps.repeat(eval_batch_size, 1).to(device)

    model.eval()

    with torch.no_grad():

        for _ in range(num_eval_ep):

            # zeros place holders
            actions = torch.zeros((eval_batch_size, max_test_ep_len, act_dim),
                                dtype=torch.float32, device=device)
            states = torch.zeros((eval_batch_size, max_test_ep_len, state_dim),
                                dtype=torch.float32, device=device)
            
            # init episode
            running_state = env.reset()
            running_reward = 0


            for t in range(max_test_ep_len):

                total_timesteps += 1

                # add state in placeholder and normalize
                states[0, t] = torch.from_numpy(running_state).to(device)
                states[0, t] = (states[0, t] - state_mean) / state_std

            
                if t < context_len:
                    _, act_preds = model.forward(timesteps[:,:context_len],
                                                states[:,:context_len],
                                                actions[:,:context_len])
                    act = act_preds[0, t].detach()
                else:
                    _, act_preds = model.forward(timesteps[:,t-context_len+1:t+1],
                                                states[:,t-context_len+1:t+1],
                                                actions[:,t-context_len+1:t+1])
                    act = act_preds[0, -1].detach()

                running_state, running_reward, done, info = env.step(act.cpu().numpy())


                # add action in placeholder
                actions[0, t] = act

                total_reward += running_reward

                if render:
                    env.render()
                if done:
                    break

    results['eval/avg_reward'] = total_reward / num_eval_ep
    results['eval/avg_ep_len'] = total_timesteps / num_eval_ep

    return results

class Dataset(Dataset):
    def __init__(self, dataset_path, context_len,online_training):
        
        rtg_scale = 10

        self.online_buffer_size=5000
        self.context_len = context_len

        # load dataset
        with open(r'/home/data_2/why_22/code/GODA//yuce/halfcheetah_medium-v2.pkl', 'rb') as f:
            self.trajectories = pickle.load(f)
        self.trajectories = self.trajectories[0:800]
        self.states, self.actions, self.rewards, self.traj_lens = [], [], [], []
        for path in self.trajectories:
            self.states.append(path['observations'])
            self.actions.append(path['actions'])
            self.traj_lens.append(len(path['observations']))
            self.rewards.append(path['rewards'])
            # path['returns_to_go'] = discount_cumsum(path['rewards'], 1.0) / rtg_scale
            # self.returns_to_go.append(path['returns_to_go'])

#normalization 
        self.states = np.concatenate(self.states, axis=0)
        self.actions = np.concatenate(self.actions, axis=0)
        self.rewards = np.concatenate(self.rewards, axis=0)
        self.states_mean, self.states_std = np.mean(self.states, axis=0), np.std(self.states, axis=0) + 1e-6
        self.actions_mean, self.actions_std = np.mean(self.actions, axis=0), np.std(self.actions, axis=0) + 1e-6
        self.rewards_mean, self.rewards_std = np.mean(self.rewards, axis=0), np.std(self.rewards, axis=0) + 1e-6
        print(f'train:statesmean and statesstd={self.states_mean} and {self.states_std}'+'\n')
        print(f'train:actions_mean and statesstd={self.actions_mean} and {self.actions_std}'+'\n')
        print(f'train:rewards_mean and statesstd={self.rewards_mean} and {self.rewards_std}'+'\n')
        for traj in self.trajectories:
            traj['observations'] = (traj['observations'] - self.states_mean) / self.states_std
            # traj['actions'] = (traj['actions'] - self.actions_mean) / self.actions_std
            traj['rewards'] = (traj['rewards'] - self.rewards_mean) / self.rewards_std

        num_timesteps = sum(self.traj_lens)
        print('=' * 50)
        print(f'Starting new experiment:  {dataset_path}')
        print(f'{len(self.traj_lens)} trajectories, {num_timesteps} timesteps found')
        print(f'Average return: {np.mean(self.rewards):.2f}, std: {np.std(self.rewards):.2f}')
        print(f'Max return: {np.max(self.rewards):.2f}, min: {np.min(self.rewards):.2f}')
        print('=' * 50)

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        traj_len = traj['observations'].shape[0]
        self.idx = idx

        if traj_len >= self.context_len:
            # sample random index to slice trajectory
            si = random.randint(0, traj_len - self.context_len)
            self.si = si

            states = torch.from_numpy(traj['observations'][si : si + self.context_len])
            actions = torch.from_numpy(traj['actions'][si : si + self.context_len])
            rewards = torch.from_numpy(traj['rewards'][si : si + self.context_len])

            # all ones since no padding

        return  states.to(torch.float32), actions.to(torch.float32), rewards.to(torch.float32)
    
    def get_index(self):
        return self.idx, self.si
class D4RLTrajectoryDataset(Dataset):
    def __init__(self, dataset_path, context_len):

        self.context_len = context_len

        # load dataset
        with open(dataset_path, 'rb') as f:
            self.trajectories = pickle.load(f)

        # calculate min len of traj, state mean and variance
        min_len = 10**6
        states,rewards = [],[]
        for traj in self.trajectories:
            traj_len = traj['observations'].shape[0]
            min_len = min(min_len, traj_len)
            states.append(traj['observations'])
            rewards.append(traj['rewards'])

        # used for input normalization
        states = np.concatenate(states, axis=0)
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        self.reward_mean, self.reward_std = np.mean(rewards, axis=0), np.std(rewards, axis=0) + 1e-6

        # normalize states
        for traj in self.trajectories:
            traj['observations'] = (traj['observations'] - self.state_mean) / self.state_std
            traj['rewards'] = (traj['rewards'] - self.reward_mean) / self.reward_std

    def get_state_stats(self):
        return self.state_mean, self.state_std,self.reward_mean, self.reward_std 

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        traj_len = traj['observations'].shape[0]

        if traj_len >= self.context_len:
            # sample random index to slice trajectory
            si = random.randint(0, traj_len - self.context_len)

            states = torch.from_numpy(traj['observations'][si : si + self.context_len])
            actions = torch.from_numpy(traj['actions'][si : si + self.context_len])
            timesteps = torch.arange(start=si, end=si+self.context_len, step=1)
            rewards = torch.from_numpy(traj['rewards'][si : si + self.context_len])

            # all ones since no padding
            traj_mask = torch.ones(self.context_len, dtype=torch.long)

        else:
            padding_len = self.context_len - traj_len

            # padding with zeros
            states = torch.from_numpy(traj['observations'])
            states = torch.cat([states,
                                torch.zeros(([padding_len] + list(states.shape[1:])),
                                dtype=states.dtype)],
                               dim=0)

            actions = torch.from_numpy(traj['actions'])
            actions = torch.cat([actions,
                                torch.zeros(([padding_len] + list(actions.shape[1:])),
                                dtype=actions.dtype)],
                               dim=0)
            rewards = torch.from_numpy(traj['rewards'])
            rewards = torch.cat([rewards,
                                torch.zeros(([padding_len] + list(rewards.shape[1:])),
                                dtype=rewards.dtype)],
                               dim=0)

            timesteps = torch.arange(start=0, end=self.context_len, step=1)


            traj_mask = torch.cat([torch.ones(traj_len, dtype=torch.long),
                                   torch.zeros(padding_len, dtype=torch.long)],
                                  dim=0)

        return  timesteps, states.to(torch.float32), actions.to(torch.float32), traj_mask, rewards.to(torch.float32)

class STARDataset(Dataset):
    def __init__(self, dataset_path, context_len, rtg_scale):

        self.context_len = context_len

        # load dataset
        with open(dataset_path, 'rb') as f:
            self.trajectories = pickle.load(f)

        # calculate min len of traj, state mean and variance
        # and returns_to_go for all traj
        min_len = 10**6
        states, actions, rewards = [],[],[]
        for traj in self.trajectories:
            traj_len = traj['observations'].shape[0]
            min_len = min(min_len, traj_len)
            states.append(traj['observations'])
            actions.append(traj['actions'])
            rewards.append(traj['rewards'])
            # calculate returns to go and rescale them
            traj['returns_to_go'] = discount_cumsum(traj['rewards'], 1.0) / rtg_scale

        # used for input normalization
        states = np.concatenate(states, axis=0)
        actions = np.concatenate(actions, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        self.action_mean, self.action_std = np.mean(actions, axis=0), np.std(actions, axis=0) + 1e-6
        self.reward_mean, self.reward_std = np.mean(rewards, axis=0), np.std(rewards, axis=0) + 1e-6

        # normalize states
        for traj in self.trajectories:
            traj['observations'] = (traj['observations'] - self.state_mean) / self.state_std
            # traj['actions'] = (traj['actions'] - self.action_mean) / self.action_std
            traj['rewards'] = (traj['rewards'] - self.reward_mean) / self.reward_std

    def get_state_stats(self):
        return self.state_mean, self.state_std, self.reward_mean, self.reward_std, self.action_mean, self.action_std

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        traj_len = traj['observations'].shape[0]

        if traj_len >= self.context_len:
            # sample random index to slice trajectory
            si = random.randint(0, traj_len - self.context_len)

            states = torch.from_numpy(traj['observations'][si : si + self.context_len])
            actions = torch.from_numpy(traj['actions'][si : si + self.context_len])
            returns_to_go = torch.from_numpy(traj['returns_to_go'][si : si + self.context_len])
            rewards = torch.from_numpy(traj['rewards'][si : si + self.context_len])
            timesteps = torch.arange(start=si, end=si+self.context_len, step=1)

            # all ones since no padding
            traj_mask = torch.ones(self.context_len, dtype=torch.long)

        else:
            padding_len = self.context_len - traj_len

            # padding with zeros
            states = torch.from_numpy(traj['observations'])
            states = torch.cat([states,
                                torch.zeros(([padding_len] + list(states.shape[1:])),
                                dtype=states.dtype)],
                               dim=0)

            actions = torch.from_numpy(traj['actions'])
            actions = torch.cat([actions,
                                torch.zeros(([padding_len] + list(actions.shape[1:])),
                                dtype=actions.dtype)],
                               dim=0)

            returns_to_go = torch.from_numpy(traj['returns_to_go'])
            returns_to_go = torch.cat([returns_to_go,
                                torch.zeros(([padding_len] + list(returns_to_go.shape[1:])),
                                dtype=returns_to_go.dtype)],
                               dim=0)
            
            rewards = torch.from_numpy(traj['rewards'])
            rewards = torch.cat([rewards,
                                torch.zeros(([padding_len] + list(rewards.shape[1:])),
                                dtype=rewards.dtype)],
                               dim=0)

            timesteps = torch.arange(start=0, end=self.context_len, step=1)

            traj_mask = torch.cat([torch.ones(traj_len, dtype=torch.long),
                                   torch.zeros(padding_len, dtype=torch.long)],
                                  dim=0)

        return  timesteps, states, actions, returns_to_go, traj_mask, rewards