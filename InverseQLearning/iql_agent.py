import numpy as np
import random
from collections import namedtuple, deque
from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim





class Agent():
    def __init__(self, state_size, action_size, action_dim, config):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = 0
        self.device = 'cuda'
        self.batch_size = 64
        self.rl = 0.005
        self.q_shift_local = QNetwork(state_size, action_size, self.seed).to(self.device)
        self.q_shift_target = QNetwork(state_size, action_size, self.seed).to(self.device)
        self.Q_local = QNetwork(state_size, action_size, self.seed).to(self.device)
        self.Q_target = QNetwork(state_size, action_size, self.seed).to(self.device)
        self.R_local = RNetwork(state_size,action_size, self.seed).to(self.device)
        self.R_target = RNetwork(state_size, action_size, self.seed).to(self.device)
        self.policy = PolicyNetwork(state_size, action_size,self.seed).to(self.device)

        # optimizer
        self.optimizer_q_shift = optim.Adam(self.q_shift_local.parameters(), lr=self.lr)
        self.optimizer_q = optim.Adam(self.Q_local.parameters(), lr=self.lr)
        self.optimizer_r = optim.Adam(self.R_local.parameters(), lr=self.lr)
        self.optimizer_p = optim.Adam(self.policy.parameters(), lr=self.lr)

    def act(self, state):
        dis, action, log_probs, ent = self.policy.sample_action(torch.Tensor(state).unsqueeze(0))
        return dis, action, log_probs, ent

    def learn(self, memory, batch_size):
        states, actions, rewards, next_states, not_dones = memory.sample(batch_size)

        # compute difference between Q_shift and y_sh
        q_sh_value = self.q_shift_local(states, actions)
        y_sh = np.empty((self.batch_size,1), dtype=np.float32)
        for idx, s in enumerate(states):
            q = []
            for action in all_actions:
                q.append(q_shift_target(s.unsqueeze(0), action.unsqueeze(0)))
            q_max = max(q)
            np.copyto(y_sh[idx], q_max.detach().numpy())


        y_sh = torch.Tensor(y_sh)
        q_shift_loss = F.mse_loss(y_sh, q_shift_values)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 


