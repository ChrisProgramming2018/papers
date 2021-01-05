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
from utils import time_format
from datetime import datetime

class TD3():
    def __init__(self, action_size, state_size, config):
        self.seed = config["seed"]
        print("TD3 seed", self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(seed=self.seed)
        random.seed(self.seed)
        self.env = gym.make(config["env_name"])
        self.env.seed(self.seed)
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
        self.env.action_space.seed(self.seed)
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
        self.vid_path = config["vid_path"]
        print("actions size ", action_size)
        print("actions min ", self.min_action)
        print("actions max ", self.max_action)
        fc1 = config["fc1_units"]
        fc2 = config["fc2_units"]
        self.actor = Actor(state_size, action_size, self.seed, fc1, fc2).to(self.device)
        self.optimizer_a = torch.optim.Adam(self.actor.parameters(), config["lr_actor"])
        self.target_actor = Actor(state_size, action_size, self.seed, fc1, fc2).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict()) 
        self.critic = QNetwork(state_size, action_size, self.seed, fc1, fc2).to(self.device)
        self.optimizer_q = torch.optim.Adam(self.critic.parameters(), config["lr_critic"])
        self.target_critic = QNetwork(state_size, action_size, self.seed, fc1, fc2).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.max_timesteps = config["max_episodes_steps"]
        self.episodes = config["episodes"]
        self.memory = ReplayBuffer((state_size, ), (action_size, ), config["buffer_size"], self.seed, self.device)
        pathname = str(config["seed"]) + str(dt_string)
        tensorboard_name = str(config["res_path"]) + '/runs/'+ "TD3" + str(pathname)
        self.writer = SummaryWriter(tensorboard_name)
        self.steps= 0
        self.actor_freq = config["actor_freq"]
        self.policy_noise = config["policy_noise"]
        self.noise_clip = config["noise_clip"]
        self.expl_noise = config["exp_noise"]

    def act(self, state):
        state  = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        action = self.actor(state.unsqueeze(0))
        actions = action.detach().cpu().numpy()[0] 
        actions = actions + + np.random.normal(0, self.max_action * self.expl_noise, size=self.action_size)
        actions = np.clip(actions, self.min_action, self.max_action)
        return actions
    
    def act_greedy(self, state):
        state  = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        action = self.actor(state.unsqueeze(0))
        actions = action.detach().cpu().numpy()[0]
        actions = np.clip(actions, self.min_action, self.max_action)
        return actions
    
    def train_agent(self):
        average_reward = 0
        scores_window = deque(maxlen=100)
        s = 0
        t0 = time.time()
        for i_epiosde in range(self.episodes):
            episode_reward = 0
            state = self.env.reset()
            for t in range(self.max_timesteps):
                s += 1
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
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
            self.writer.add_scalar('steps_in_episode', t, self.steps)
            
    
    def learn(self):
        self.steps += 1
        states, rewards, actions, next_states, dones = self.memory.sample(self.batch_size)
        with torch.no_grad():
            next_action = self.target_actor(next_states)
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            print(noise)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
            q1_target, q2_target = self.target_critic(next_states, next_action)
            q_target = torch.min(q1_target, q2_target)
            q_target = rewards + (self.gamma * q_target * (1 - dones))
        q_pre1, q_pre2 = self.critic(states, actions)
        loss = F.mse_loss(q_pre1, q_target) + F.mse_loss(q_pre2, q_target)
        self.writer.add_scalar('Q_loss', loss, self.steps)
        self.optimizer_q.zero_grad()
        loss.backward()
        self.optimizer_q.step()
        # delay actor update
        if self.steps % self.actor_freq == 0:
            #-------------------------------update-actor-------------------------------------------------
            actor_actions  = self.actor(states)
            q_values = self.critic.Q1(states, actor_actions)
            loss_actor = -q_values.mean()
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
        env  = wrappers.Monitor(self.env, str(self.vid_path) + "/{}".format(self.steps), video_callable=lambda episode_id: True,force=True)
        average_reward = 0
        scores_window = deque(maxlen=100)
        for i_epiosde in range(eval_episodes):
            print("Eval Episode {} of {} ".format(i_epiosde, self.episodes))
            episode_reward = 0
            state = env.reset()
            while True: 
                action = self.act_greedy(state)
                state, reward, done, _ = env.step(action)
                episode_reward += reward
                if done:
                    scores_window.append(episode_reward)
                    break
        average_reward = np.mean(scores_window)
        self.writer.add_scalar('Eval_reward', average_reward, self.steps)
