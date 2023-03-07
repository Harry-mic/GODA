import random
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
import pdb

class Dataset(Dataset):
    def __init__(self, dataset_path, context_len,online_training):

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
            self.rewards.append(path['rewards'])
            self.traj_lens.append(len(path['observations']))
#normalization 
        self.states = np.concatenate(self.states, axis=0)
        self.actions = np.concatenate(self.actions, axis=0)
        self.rewards = np.concatenate(self.rewards, axis=0)
        self.states_mean, self.states_std = np.mean(self.states, axis=0), np.std(self.states, axis=0) + 1e-6
        self.actions_mean, self.actions_std = np.mean(self.actions, axis=0), np.std(self.actions, axis=0) + 1e-6
        self.rewards_mean, self.rewards_std = np.mean(self.rewards, axis=0), np.std(self.rewards, axis=0) + 1e-6
        for traj in self.trajectories:
            traj['observations'] = (traj['observations'] - self.states_mean) / self.states_std
            traj['actions'] = (traj['actions'] - self.actions_mean) / self.actions_std
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

        if traj_len >= self.context_len:
            # sample random index to slice trajectory
            si = random.randint(0, traj_len - self.context_len)

            states = torch.from_numpy(traj['observations'][si : si + self.context_len])
            actions = torch.from_numpy(traj['actions'][si : si + self.context_len])
            rewards = torch.from_numpy(traj['rewards'][si : si + self.context_len])
            # timesteps = torch.arange(start=si, end=si+self.context_len, step=1)

            # all ones since no padding

        return  states.to(torch.float32), actions.to(torch.float32), rewards.to(torch.float32)

class TESTDataset(Dataset):
    def __init__(self, dataset_path, context_len,online_training):

        self.online_buffer_size=5000
        self.context_len = context_len

        # load dataset
        with open(r'/home/data_2/why_22/code/GODA//yuce/halfcheetah_medium-v2.pkl', 'rb') as f:
            self.trajectories = pickle.load(f)
        self.trajectories = self.trajectories[801:1000]
        self.states, self.actions, self.rewards, self.traj_lens = [], [], [], []
        for path in self.trajectories:
            self.states.append(path['observations'])
            self.actions.append(path['actions'])
            self.rewards.append(path['rewards'])
            self.traj_lens.append(len(path['observations']))
#normalization 
        self.states = np.concatenate(self.states, axis=0)
        self.actions = np.concatenate(self.actions, axis=0)
        self.rewards = np.concatenate(self.rewards, axis=0)
        self.states_mean, self.states_std = np.mean(self.states, axis=0), np.std(self.states, axis=0) + 1e-6
        self.actions_mean, self.actions_std = np.mean(self.actions, axis=0), np.std(self.actions, axis=0) + 1e-6
        self.rewards_mean, self.rewards_std = np.mean(self.rewards, axis=0), np.std(self.rewards, axis=0) + 1e-6
        print(f'test:statesmean and statesstd={self.states_mean} and {self.states_std}'+'\n')
        print(f'test:actions_mean and statesstd={self.actions_mean} and {self.actions_std}'+'\n')
        print(f'test:rewards_mean and statesstd={self.rewards_mean} and {self.rewards_std}'+'\n')
        for traj in self.trajectories:
            traj['observations'] = (traj['observations'] - self.states_mean) / self.states_std
            traj['actions'] = (traj['actions'] - self.actions_mean) / self.actions_std
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

        if traj_len >= self.context_len:
            # sample random index to slice trajectory
            si = random.randint(0, traj_len - self.context_len)

            states = torch.from_numpy(traj['observations'][si : si + self.context_len])
            actions = torch.from_numpy(traj['actions'][si : si + self.context_len])
            rewards = torch.from_numpy(traj['rewards'][si : si + self.context_len])
            # timesteps = torch.arange(start=si, end=si+self.context_len, step=1)

            # all ones since no padding

        return  states.to(torch.float32), actions.to(torch.float32), rewards.to(torch.float32)