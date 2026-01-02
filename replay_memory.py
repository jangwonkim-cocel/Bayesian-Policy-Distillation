import torch
import numpy as np
import copy


class ReplayMemory:
    def __init__(self, state_dim, action_dim, device='cuda', capacity=5e6):
        self.capacity = int(capacity)
        self.size = 0
        self.position = 0

        self.state_buffer = np.empty(shape=(self.capacity, state_dim), dtype=np.float32)
        self.action_buffer = np.empty(shape=(self.capacity, action_dim), dtype=np.float32)
        self.reward_buffer = np.empty(shape=(self.capacity, 1), dtype=np.float32)
        self.next_state_buffer = np.empty(shape=(self.capacity, state_dim), dtype=np.float32)
        self.done_buffer = np.empty(shape=(self.capacity, 1), dtype=np.float32)

    def normalize_states(self, eps=1e-3):
        mean = np.mean(copy.deepcopy(self.state_buffer).astype('float64'), axis=0)
        std = np.std(copy.deepcopy(self.state_buffer).astype('float64'), axis=0) + eps
        self.state_buffer = (self.state_buffer.astype('float64') - mean) / std
        self.next_state_buffer = (self.next_state_buffer.astype('float64') - mean) / std

        self.state_buffer = self.state_buffer.astype('float32')
        self.next_state_buffer = self.next_state_buffer.astype('float32')
        return

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)

        states = torch.FloatTensor(self.state_buffer[idxs]).to('cuda')
        actions = torch.FloatTensor(self.action_buffer[idxs]).to('cuda')
        rewards = torch.FloatTensor(self.reward_buffer[idxs]).to('cuda')
        next_states = torch.FloatTensor(self.next_state_buffer[idxs]).to('cuda')
        dones = torch.FloatTensor(self.done_buffer[idxs]).to('cuda')

        return states, actions, rewards, next_states, dones


