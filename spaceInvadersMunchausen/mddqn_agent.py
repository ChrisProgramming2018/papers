import numpy as np
import random
from collections import namedtuple, deque
from model import QNetwork , Encoder
from replay_buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import gym
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
import torch.optim as optim
import sys
from utils import time_format
from framestack import FrameStack
from datetime import datetime
import time


class MDDQNAgent():
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
        self.seed = config["seed"]
        torch.manual_seed(self.seed)
        np.random.seed(seed=self.seed)
        random.seed(self.seed)
        env = gym.make(config["env_name"])
        self.env = FrameStack(env, config)
        self.env.action_space.seed(self.seed)
        self.action_size = action_size
        self.seed = int(config["seed"])
        self.lr = config['lr']
        self.batch_size = config['batch_size']
        self.device = config['device']
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.train_freq = config['train_freq']
        self.total_frames = int(config['total_frames'])
        self.start_timesteps = int(config['start_timesteps'])
        self.eval = config["eval"]
        obs_shape = (config["history_length"], config["size"], config["size"])
        self.replay_buffer = ReplayBuffer(obs_shape, (1, ), int(config["buffer_size"]), self.seed, config["image_pad"], config['device'])
        self.qnetwork_local = QNetwork(state_size, action_size, self.seed).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size, self.seed).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)
        self.encoder = Encoder(config).to(self.device)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), self.lr)
        self.t_step = 0
        self.entropy = 0.03
        self.alpha_m = 0.9
        self.clip_log = -1
        self.eps_decay = config["eps_decay"]
        self.eps_end = config["eps_min"]
        self.all_actions = []
        now = datetime.now()
        self.vid_path = "vid"
        dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
        pathname = dt_string + "seed_" + str(config['seed'])
        tensorboard_name = 'runs/' + pathname
        self.writer = SummaryWriter(tensorboard_name)
        for a in range(self.action_size):
            action = torch.Tensor([1 for i in range(self.batch_size)]).type(torch.long) * 0 +  a
            self.all_actions.append(action.to(self.device))
    
    def step(self):
        self.t_step +=1 
        if self.t_step % self.train_freq == 0:
            self.learn()

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """

        # Epsilon-greedy action selection
        if random.random() > eps:
            self.qnetwork_local.eval()
            with torch.no_grad():
                state = torch.from_numpy(state).unsqueeze(0).to(self.device)
                state = state.type(torch.float32).div_(255)
                state = self.encoder.create_vector(state) 
                action_values = self.qnetwork_local(state)
                self.qnetwork_local.train()
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Get max predicted Q values (for next states) from target model
        #local_actions = self.qnetwork_local(next_states).detach().max(1)[0]
        #Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, local_actions)
        states = states.type(torch.float32).div_(255)
        states = self.encoder.create_vector(states) 
        next_states = next_states.type(torch.float32).div_(255)
        next_states = self.encoder.create_vector(next_states)
        q_values_next = self.qnetwork_target(next_states).detach()
        q_values_next_action = self.qnetwork_local(next_states).detach()
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
        self.encoder_optimizer.zero_grad()
        loss.backward()
        self.encoder_optimizer.step()
        self.optimizer.step()
        self.writer.add_scalar('loss', loss, self.t_step)

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

    def train(self):
        
        scores_window = deque(maxlen=100)
        step_window = deque(maxlen=100)
        eps = 1
        t0 = time.time()
        total_timesteps = 0
        i_episode = 0
        total_timesteps = 0
        while total_timesteps < self.total_frames:
            state = self.env.reset()
            env_score = 0
            steps = 0
            while True:
                total_timesteps += 1
                steps += 1
                action = self.act(state, eps)
                next_state, reward, done, _ = self.env.step(action)
                eps = max(self.eps_end, self.eps_decay*eps) # decrease epsilon
                if self.start_timesteps < total_timesteps:
                    self.step()
                env_score += reward
                self.replay_buffer.add(state, action, reward, next_state, done, done)
                state = next_state
                
                if done:
                    i_episode += 1
                    break
            
            scores_window.append(env_score)       # save most recent score
            step_window.append(steps)       # save most recent score
            mean_reward = np.mean(scores_window)
            mean_steps = np.mean(step_window)
            self.writer.add_scalar('env_reward', env_score, total_timesteps)
            self.writer.add_scalar('mean_reward', mean_reward, total_timesteps)
            self.writer.add_scalar('mean_steps', mean_steps, total_timesteps)
            self.writer.add_scalar('steps', steps, total_timesteps)
            print(' Totalsteps {} Episode {} Step {} Reward {} Average Score: {:.2f} epsilon {:.2f} time {}'  .format(total_timesteps, i_episode, steps, env_score, np.mean(scores_window), eps, time_format(time.time()-t0)))
            if i_episode % self.eval == 0:

                print('\rEpisode {}\tAverage Score: {:.2f} Time: {}'.format(i_episode, np.mean(scores_window),  time_format(time.time()-t0)))
   
   def eval_policy(self, eval_episode=4):
       env  = wrappers.Monitor(self.env, str(self.vid_path) + "/{}".format(self.t_step), video_callable=lambda episode_id: True,force=True)
       average_reward = 0
       scores_window = deque(maxlen=eval_episodes)
       for i_epiosde in range(eval_episodes):
           print("Eval Episode {} of {} ".format(i_epiosde, eval_episodes))
           episode_reward = 0
           state = env.reset()
           while True:
               action = self.act(state)
               state, reward, done, _ = env.step(action)
               episode_reward += reward
               if done:
                   break
            scores_window.append(episode_reward)
        average_reward = np.mean(scores_window)
        writer.add_scalar('Eval_reward', average_reward, self.t_step)
