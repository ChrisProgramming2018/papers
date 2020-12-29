import os
import time
import random
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import wrappers
from torch.autograd import Variable
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple, deque
from models import QNetwork, SACActor
from replay_buffer import ReplayBuffer
from utils import time_format
from torch.optim import Adam


from model import GaussianPolicy

class SACAgent():
    def __init__(self, action_size, state_size, config):
        self.action_size = action_size
        self.state_size = state_size
        self.min_action = config["min_action"]
        self.max_action = config["max_action"]
        self.seed = config["seed"]
        self.tau = config["tau"]
        self.gamma = config["gamma"]
        self.batch_size = config["batch_size"]
        if not torch.cuda.is_available():
            config["device"] == "cpu"
        self.device = config["device"]
        self.eval = config["eval"]
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.vid_path = config["vid_path"]
        print("actions size ", action_size)
        print("actions min ", self.min_action)
        print("actions max ", self.max_action)
        self.critic = QNetwork(state_size, action_size, config["fc1_units"], config["fc2_units"]).to(self.device)
        self.q_optim = torch.optim.Adam(self.critic.parameters(), config["lr_critic"])
        self.target_critic = QNetwork(state_size, action_size, config["fc1_units"], config["fc2_units"]).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = Adam([self.log_alpha], lr=config["lr_alpha"])
        #self.policy = SACActor(state_size, action_size).to(self.device)
        self.policy = GaussianPolicy(state_size, action_size, 256).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=config["lr_policy"])
        self.max_timesteps = config["max_episodes_steps"]
        self.episodes = config["episodes"]
        self.memory = ReplayBuffer((state_size, ), (action_size, ), config["buffer_size"], self.device)
        pathname = config["seed"]
        tensorboard_name = str(config["res_path"]) + '/runs/' + str(pathname)
        self.writer = SummaryWriter(tensorboard_name)
        self.steps= 0
        self.target_entropy = -torch.prod(torch.Tensor(action_size).to(self.device)).item()

    def act(self, state, evaluate=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            if evaluate is False:
                action, _, _ = self.policy.sample(state)
            else:
                _, _, action = self.policy.sample(state)
            # action = np.clip(action, self.min_action, self.max_action)
            action = action.cpu().numpy()[0]
        #print(action)
        return action
    
    def train_agent(self):
        env = gym.make("LunarLanderContinuous-v2")
        average_reward = 0
        scores_window = deque(maxlen=100)
        s = 0
        t0 = time.time()
        for i_epiosde in range(self.episodes):
            episode_reward = 0
            state = env.reset()
            for t in range(self.max_timesteps):
                s += 1
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                if i_epiosde > 3:
                    self.learn()
                self.memory.add(state, reward, action, next_state, done)
                state = next_state
                if done:
                    scores_window.append(episode_reward)
                    break
            if i_epiosde % self.eval == 0:
                self.eval_policy()
            ave_reward = np.mean(scores_window)
            print("Epiosde {} Steps {} Reward {} Reward averge{} Time {}".format(i_epiosde, t, episode_reward, np.mean(scores_window), time_format(time.time() - t0)))
            self.writer.add_scalar('Aver_reward', ave_reward, self.steps)
            
    
    def learn(self):
        self.steps += 1
        states, rewards, actions, next_states, dones = self.memory.sample(self.batch_size)
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_states)
            target_q1, target_q2 = self.target_critic(next_states, next_state_action)
            target_min = torch.min(target_q1, target_q2)
            q_target = target_min - (self.alpha * next_state_log_pi)
            next_q_value = rewards + (1 - dones) * self.gamma * q_target
        
        qf1, qf2 = self.critic(states, actions)
        # --------------------------update-q--------------------------------------------------------
        loss = F.mse_loss(qf1, next_q_value) + F.mse_loss(qf2, next_q_value) 
        self.q_optim.zero_grad() 
        loss.backward()
        self.q_optim.step()
        self.writer.add_scalar('loss/q', loss, self.steps)


        # --------------------------update-policy--------------------------------------------------------
        pi, log_pi, _ = self.policy.sample(states)
        q_pi1, q_pi2 = self.critic(states, pi)
        min_q_values = torch.min(q_pi1, q_pi2)
        policy_loss = ((self.alpha * log_pi) - min_q_values).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        self.writer.add_scalar('loss/policy', policy_loss, self.steps)
        
        # --------------------------update-alpha--------------------------------------------------------
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.writer.add_scalar('loss/alpha', alpha_loss, self.steps)

        self.soft_udapte(self.critic, self.target_critic)
        self.alpha = self.log_alpha.exp()


    
    def soft_udapte(self, online, target):
        for param, target_parm in zip(online.parameters(), target.parameters()):
            target_parm.data.copy_(self.tau * param.data + (1 - self.tau) * target_parm.data)

    def eval_policy(self, eval_episodes=4):
        env = gym.make("LunarLanderContinuous-v2")
        # env  = wrappers.Monitor(env, str(self.vid_path) + "/{}".format(self.steps), video_callable=lambda episode_id: True,force=True)
        average_reward = 0
        scores_window = deque(maxlen=100)
        for i_epiosde in range(eval_episodes):
            print("Eval Episode {} of {} ".format(i_epiosde, eval_episodes))
            episode_reward = 0
            state = env.reset()
            while True: 
                action = self.act(state, evaluate=True)
                state, reward, done, _ = env.step(action)
                episode_reward += reward
                if done:
                    break
            scores_window.append(episode_reward)
        average_reward = np.mean(scores_window)
        self.writer.add_scalar('Eval_reward', average_reward, self.steps)
