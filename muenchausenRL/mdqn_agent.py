import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim
import sys


class MDQNAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, config):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = int(config["seed"])
        self.lr = config['lr']
        self.batch_size = config['batch_size']
        self.device = config['device']
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.train_freq = config['train_freq']
        print("seed", self.seed)
        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, self.seed).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size, self.seed).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)
        self.t_step = 0
        self.entropy = 0.03
        self.alpha_m = 0.9
        self.clip_log = -1
        self.all_actions = []
        for a in range(self.action_size):
            action = torch.Tensor([1 for i in range(self.batch_size)]).type(torch.long) * 0 +  a
            self.all_actions.append(action.to(self.device))
    def step(self, memory):
        self.t_step +=1 
        if self.t_step % self.train_freq == 0:
            if len(memory) > self.batch_size:
                self.learn(memory)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, memory):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = memory.sample(self.batch_size)

        # Get max predicted Q values (for next states) from target model
        #local_actions = self.qnetwork_local(next_states).detach().max(1)[0]
        #Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, local_actions)
        q_values_next = self.qnetwork_target(next_states).detach()
        prob_next_state = F.softmax(q_values_next, dim=1)
        Q_targets_next = 0
        for  action in self.all_actions:
            action_prob = prob_next_state.gather(1, action.unsqueeze(1))
            action_prob = action_prob + torch.finfo(torch.float32).eps
            log_action_prob = torch.log(action_prob)
            log_action_prob = torch.clamp(log_action_prob, min= self.clip_log, max=0)
            soft_target = self.entropy * log_action_prob
            q_values = q_values_next.gather(1, action.unsqueeze(1))
            Q_targets_next = Q_targets_next + (action_prob * (q_values - soft_target))
     
        # red part log prob of action
        q_values = self.qnetwork_target(states)
        output = F.softmax(q_values, dim=1)
        action_prob = output.gather(1, actions)
        action_prob = action_prob + torch.finfo(torch.float32).eps
        action_prob = torch.log(action_prob)
        action_prob = torch.clamp(action_prob, min= self.clip_log, max=0)
        extend = self.entropy * self.alpha_m * action_prob
        # Compute Q targets for current states 
        Q_targets = rewards + extend + (self.gamma * Q_targets_next * dones)

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets.detach())
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
