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
from models import Actor, QNetwork
from replay_buffer import ReplayBuffer
from utils import OrnsteinUhlenbeckProcess, time_format


class DDPGAgent():
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
        self.actor = Actor(state_size, action_size, config["fc1_units"], config["fc2_units"]).to(self.device)
        self.optimizer_a = torch.optim.Adam(self.actor.parameters(), config["lr_actor"])
        self.target_actor = Actor(state_size, action_size, config["fc1_units"], config["fc2_units"]).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict()) 
        self.critic = QNetwork(state_size, action_size, config["fc1_units"], config["fc2_units"]).to(self.device)
        self.optimizer_q = torch.optim.Adam(self.critic.parameters(), config["lr_critic"])
        self.target_critic = QNetwork(state_size, action_size, config["fc1_units"], config["fc2_units"]).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.noise = OrnsteinUhlenbeckProcess(mu=np.zeros(action_size), dimension=action_size)
        self.max_timesteps = config["max_episodes_steps"]
        self.noise.reset()
        self.episodes = config["episodes"]
        self.memory = ReplayBuffer((state_size, ), (action_size, ), config["buffer_size"], self.device)
        pathname = config["seed"]
        tensorboard_name = str(config["res_path"]) + '/runs/' + str(pathname)
        self.writer = SummaryWriter(tensorboard_name)
        self.steps= 0

    def act(self, state):
        state  = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        action = self.actor(state.unsqueeze(0))
        noise =  self.noise.step()
        actions = action.detach().cpu().numpy()[0] + noise
        actions = np.clip(actions, self.min_action, self.max_action)
        return actions
    
    def act_greedy(self, state):
        state  = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        action = self.actor(state.unsqueeze(0))
        actions = action.detach().cpu().numpy()[0]
        actions = np.clip(actions, self.min_action, self.max_action)
        return actions
    
    def train_agent(self):
        env = gym.make("LunarLanderContinuous-v2")
        average_reward = 0
        scores_window = deque(maxlen=100)
        s = 0
        t0 = time.time()
        for i_epiosde in range(self.episodes):
            episode_reward = 0
            state = env.reset()
            self.noise.reset()
            for t in range(self.max_timesteps):
                s += 1
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                if i_epiosde > 10:
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
            next_action = self.target_actor(next_states)
            q_target = self.target_critic(next_states, next_action)
            q_target = rewards + (self.gamma * q_target * (1 - dones))
        q_pre = self.critic(states, actions)
        loss = F.mse_loss(q_pre, q_target)
        
        self.writer.add_scalar('Q_loss', loss, self.steps)
        self.optimizer_q.zero_grad()
        loss.backward()
        self.optimizer_q.step()
        
        #-------------------------------update-actor-------------------------------------------------
        actor_actions  = self.actor(states)
        q_values = self.critic(states, actor_actions)
        loss_actor = - q_values.mean()
        self.optimizer_a.zero_grad()
        loss_actor.backward()
        self.writer.add_scalar('Actor_loss', loss_actor, self.steps)
        self.optimizer_a.step()
        #-------------------------------update-networks-------------------------------------------------
        self.soft_udapte(self.critic, self.target_critic)
        self.soft_udapte(self.actor, self.target_actor)
    
    
    def soft_udapte(self, online, target):
        for param, target_parm in zip(online.parameters(), target.parameters()):
            target_parm.data.copy_(self.tau * param.data + (1 - self.tau) * target_parm.data)




    def eval_policy(self, eval_episodes=4):
        env = gym.make("LunarLanderContinuous-v2")
        env  = wrappers.Monitor(env, str(self.vid_path) + "/{}".format(self.steps), video_callable=lambda episode_id: True,force=True)
        average_reward = 0
        scores_window = deque(maxlen=100)
        s = 0
        for i_epiosde in range(eval_episodes):
            print("Eval Episode {} of {} ".format(i_epiosde, self.episodes))
            episode_reward = 0
            state = env.reset()
            while True: 
                s += 1
                action = self.act_greedy(state)
                state, reward, done, _ = env.step(action)
                episode_reward += reward
                if done:
                    scores_window.append(episode_reward)
                    break
        average_reward = np.mean(scores_window)
        self.writer.add_scalar('Eval_reward', average_reward, self.steps)
