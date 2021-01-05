import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, state_shape, action_size, capacity, seed, device):
        np.random.seed(seed=seed)
        self.seed = torch.manual_seed(seed)
        self.state_size = state_shape
        self.action_size = action_size
        self.capacity = capacity
        self.device = device
        self.states = np.empty((capacity, *state_shape), dtype=np.float32)
        self.next_states = np.empty((capacity, *state_shape), dtype=np.float32)
        self.actions = np.empty((capacity, *action_size), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=np.int8)
        self.idx = 0
        self.full = False

    def add(self, state, reward, action, next_state, done):
        np.copyto(self.states[self.idx], state)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_states[self.idx], next_state)
        np.copyto(self.dones[self.idx],  done)
        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def sample(self, batch_size):
        ids = np.random.randint(0, self.capacity if self.full else self.idx, size=batch_size)
        states = torch.as_tensor(self.states[ids], device=self.device)
        next_states = torch.as_tensor(self.next_states[ids], device=self.device)
        actions = torch.as_tensor(self.actions[ids], device=self.device)
        rewards = torch.as_tensor(self.rewards[ids], device=self.device)
        dones = torch.as_tensor(self.dones[ids], device=self.device)
        return states, rewards, actions, next_states, dones

    def save_memory(self,):
        pass

    def load_memory(self,):
        pass
