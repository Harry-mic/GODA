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
        print(f'train:statesmean and statesstd={self.states_mean} and {self.states_std}'+'\n')
        print(f'train:actions_mean and statesstd={self.actions_mean} and {self.actions_std}'+'\n')
        print(f'train:rewards_mean and statesstd={self.rewards_mean} and {self.rewards_std}'+'\n')
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
        self.idx = idx

        if traj_len >= self.context_len:
            # sample random index to slice trajectory
            si = random.randint(0, traj_len - self.context_len)
            self.si = si

            states = torch.from_numpy(traj['observations'][si : si + self.context_len])
            actions = torch.from_numpy(traj['actions'][si : si + self.context_len])
            rewards = torch.from_numpy(traj['rewards'][si : si + self.context_len])
            # timesteps = torch.arange(start=si, end=si+self.context_len, step=1)

            # all ones since no padding

        return  states.to(torch.float32), actions.to(torch.float32), rewards.to(torch.float32)
    
    def get_index(self):
        return self.idx, self.si

class TESTDataset(Dataset):
    def __init__(self, dataset_path, context_len,online_training):

        self.online_buffer_size=5000
        self.context_len = context_len

        # load dataset
        with open(r'/home/data_2/why_22/code/GODA//yuce/halfcheetah_medium-v2.pkl', 'rb') as f:
            self.trajectories = pickle.load(f)
        self.trajectoriess = self.trajectories[801:1000]
        self.statess, self.actionss, self.rewardss, self.traj_lenss = [], [], [], []
        for path in self.trajectoriess:
            self.statess.append(path['observations'])
            self.actionss.append(path['actions'])
            self.rewardss.append(path['rewards'])
            self.traj_lenss.append(len(path['observations']))
#normalization 
        self.statess = np.concatenate(self.statess, axis=0)
        self.actionss = np.concatenate(self.actionss, axis=0)
        self.rewardss = np.concatenate(self.rewardss, axis=0)
        self.statess_mean, self.statess_std = np.mean(self.statess, axis=0), np.std(self.statess, axis=0) + 1e-6
        self.actionss_mean, self.actionss_std = np.mean(self.actionss, axis=0), np.std(self.actionss, axis=0) + 1e-6
        self.rewardss_mean, self.rewardss_std = np.mean(self.rewardss, axis=0), np.std(self.rewardss, axis=0) + 1e-6
        print(f'test:statesmean and statesstd={self.statess_mean} and {self.statess_std}'+'\n')
        print(f'test:actions_mean and statesstd={self.actionss_mean} and {self.actionss_std}'+'\n')
        print(f'test:rewards_mean and statesstd={self.rewardss_mean} and {self.rewardss_std}'+'\n')
        for traj in self.trajectoriess:
            traj['observations'] = (traj['observations'] - self.statess_mean) / self.statess_std
            traj['actions'] = (traj['actions'] - self.actionss_mean) / self.actionss_std
            traj['rewards'] = (traj['rewards'] - self.rewardss_mean) / self.rewardss_std

        num_timestepss = sum(self.traj_lenss)
        print('=' * 50)
        print(f'Starting new experiment:  {dataset_path}')
        print(f'{len(self.traj_lenss)} trajectories, {num_timestepss} timesteps found')
        print(f'Average return: {np.mean(self.rewardss):.2f}, std: {np.std(self.rewardss):.2f}')
        print(f'Max return: {np.max(self.rewardss):.2f}, min: {np.min(self.rewardss):.2f}')
        print('=' * 50)

    def __len__(self):
        return len(self.trajectoriess)

    def __getitem__(self, idx):
        traj = self.trajectoriess[idx]
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
    


class StarrDataset(Dataset):
    def __init__(self, dataset_path, context_len,online_training):

        self.online_buffer_size=5000
        self.context_len = context_len

        # load dataset
        with open(r'/home/data_2/why_22/code/GODA//yuce/halfcheetah_medium-v2.pkl', 'rb') as f:
            self.trajectories = pickle.load(f)
        self.trajectoriess = self.trajectories
        self.statess, self.actionss, self.rewardss, self.traj_lenss = [], [], [], []
        for path in self.trajectoriess:
            self.statess.append(path['observations'])
            self.actionss.append(path['actions'])
            self.rewardss.append(path['rewards'])
            self.traj_lenss.append(len(path['observations']))
#normalization 
        self.statess = np.concatenate(self.statess, axis=0)
        self.actionss = np.concatenate(self.actionss, axis=0)
        self.rewardss = np.concatenate(self.rewardss, axis=0)
        self.statess_mean, self.statess_std = np.mean(self.statess, axis=0), np.std(self.statess, axis=0) + 1e-6
        self.actionss_mean, self.actionss_std = np.mean(self.actionss, axis=0), np.std(self.actionss, axis=0) + 1e-6
        self.rewardss_mean, self.rewardss_std = np.mean(self.rewardss, axis=0), np.std(self.rewardss, axis=0) + 1e-6
        # print(f'test:statesmean and statesstd={self.statess_mean} and {self.statess_std}'+'\n')
        # print(f'test:actions_mean and statesstd={self.actionss_mean} and {self.actionss_std}'+'\n')
        # print(f'test:rewards_mean and statesstd={self.rewardss_mean} and {self.rewardss_std}'+'\n')
        for traj in self.trajectoriess:
            traj['observations'] = (traj['observations'] - self.statess_mean) / self.statess_std
            traj['actions'] = (traj['actions'] - self.actionss_mean) / self.actionss_std
            traj['rewards'] = (traj['rewards'] - self.rewardss_mean) / self.rewardss_std

        # num_timestepss = sum(self.traj_lenss)
        # print('=' * 50)
        # print(f'Starting new experiment:  {dataset_path}')
        # print(f'{len(self.traj_lenss)} trajectories, {num_timestepss} timesteps found')
        # print(f'Average return: {np.mean(self.rewardss):.2f}, std: {np.std(self.rewardss):.2f}')
        # print(f'Max return: {np.max(self.rewardss):.2f}, min: {np.min(self.rewardss):.2f}')
        # print('=' * 50)

    def __len__(self):
        return len(self.trajectoriess)

    def __getitem__(self, idx):
        traj = self.trajectoriess[idx+600]
        traj_len = traj['observations'].shape[0]
        self.idx = idx

        if traj_len >= self.context_len:
            # sample random index to slice trajectory
            si = 500
            self.si = si
            #si = random.randint(0, traj_len - self.context_len)

            states = torch.from_numpy(traj['observations'][si : si + self.context_len])
            actions = torch.from_numpy(traj['actions'][si : si + self.context_len])
            rewards = torch.from_numpy(traj['rewards'][si : si + self.context_len])
            # timesteps = torch.arange(start=si, end=si+self.context_len, step=1)

            # all ones since no padding

        return  states.to(torch.float32), actions.to(torch.float32), rewards.to(torch.float32)
    
    def get_index(self):
        return self.idx, self.si