import numpy as np
import random
from collections import namedtuple, deque

from model import TwinnedQNetwork, CateoricalPolicy
from utils import disable_gradients
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn.functional as F
from torch.optim import Adam

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        lr = LR
        self.summary_dir = 'runs/sac'
        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.device = "cuda"
        target_entropy_ratio = 0.96
        dueling_net = False
        self.batch_size = 64
        # Q-Network
        self.policy = CateoricalPolicy(state_size, action_size).to(self.device)

        self.online_critic = TwinnedQNetwork(state_size, action_size, dueling_net = dueling_net).to(device=self.device)

        self.target_critic = TwinnedQNetwork(state_size, action_size, dueling_net = dueling_net).to(device=self.device).eval()

        # Copy parameters of the learning network to the target network.

        self.target_critic.load_state_dict(self.online_critic.state_dict())
        # Disable gradient calculations of the target network.
        disable_gradients(self.target_critic)
        self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        self.q1_optim = Adam(self.online_critic.Q1.parameters(), lr=lr)
        self.q2_optim = Adam(self.online_critic.Q2.parameters(), lr=lr)
        # Target entropy is -log(1/|A|) * ratio (= maximum entropy * ratio).
        self.target_entropy = - np.log(1.0 / action_size) * target_entropy_ratio
        # We optimize log(alpha), instead of alpha.
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = Adam([self.log_alpha], lr=lr)
        self.gamma = 0.99
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.total_steps =0

    def explore(self, state):
        # Act with randomness.
        state = torch.ByteTensor(state[None, ...]).to(self.device).float() / 255.
        with torch.no_grad():
            action, _, _ = self.policy.sample(state)
        return action.item()

    def exploit(self, state):
        # Act without randomness.
        state = torch.ByteTensor(
        state[None, ...]).to(self.device).float() / 255.
        with torch.no_grad():
            action = self.policy.act(state)
        return action.item()



    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            self.total_steps += 1
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                experiences2 = self.memory.sample()
                self.learn(experiences, experiences2)

    def act(self, state):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        return self.explore(state)



    def learn(self, experiences1, experinces2):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences1

        curr_q1, curr_q2 = self.online_critic(states)
        curr_q1 = curr_q1.gather(1, actions.long())
        curr_q2 = curr_q2.gather(1, actions.long())
        #print("update weights")
        # compute target
        with torch.no_grad():
            _, action_probs, log_action_probs = self.policy.sample(next_states)
            next_q1, next_q2 = self.target_critic(next_states)
            next_q = (action_probs * (torch.min(next_q1, next_q2) - self.alpha * log_action_probs)).sum(dim=1, keepdim=True)



        target_q = rewards + (1.0 - dones) * self.gamma * next_q

        # We log means of Q to monitor training.
        #mean_q1 = curr_q1.detach().mean().item()
        #mean_q2 = curr_q2.detach().mean().item()

        # Critic loss is mean squared TD errors with priority weights.
        q1_loss = torch.mean((curr_q1 - target_q).pow(2))
        q2_loss = torch.mean((curr_q2 - target_q).pow(2))
        self.writer.add_scalar( 'loss/Q1', q1_loss.detach().item(), self.total_steps)

        policy_loss, entropies = self.update_policy(experinces2)
        entropy_loss = self.entropy_loss(entropies)



        self.update_params(self.q1_optim, q1_loss)
        self.update_params(self.q2_optim, q2_loss)
        self.update_params(self.policy_optim, policy_loss)
        self.update_params(self.alpha_optim, entropy_loss)
        self.alpha = self.log_alpha.exp()
        self.soft_update(self.online_critic, self.target_critic, TAU)

    def update_policy(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        _, action_probs, log_action_probs = self.policy.sample(states.detach())
        with torch.no_grad():
            q1, q2 = self.online_critic(states)
            q = torch.min(q1, q2)
        # Expectations of entropies.
        entropies = -torch.sum(action_probs * log_action_probs, dim = 1, keepdim = True)
        # Expectations of Q.
        q = torch.sum(torch.min(q1, q2) * action_probs, dim=1, keepdim=True)
        # Policy objective is maximization of (Q + alpha * entropy)
        policy_loss = (- q - self.alpha * entropies).mean()
        return policy_loss, entropies.detach()

    def entropy_loss(self, entropies):
        assert not entropies.requires_grad
        entropy_loss = -torch.mean(self.log_alpha * (self.target_entropy - entropies))
        return entropy_loss


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
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def update_params(self, optim, loss, retain_graph=False):
        optim.zero_grad()
        loss.backward(retain_graph=retain_graph)
        optim.step()


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to \
            (device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to \
            (device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)