import gym
import random
import torch
import numpy as np
from mdqn_agent import MDQNAgent
from mddqn_agent import MDDQNAgent
from replay_buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import argparse
import json
import time
from datetime import datetime
from collections import namedtuple, deque
import os

def time_format(sec):
    """

    Args:
        param1():
    """
    hours = sec // 3600
    rem = sec - hours * 3600
    mins = rem // 60
    secs = rem - mins * 60
    return hours, mins, round(secs,2)


def main(args):
    with open (args.param, "r") as f:
        config = json.load(f)
    env = gym.make('LunarLander-v2')
    config["seed"] = args.seed
    path = config["locexp"]
    if not os.path.exists(path):
        os.makedirs(path)
    #env = gym.wrappers.Monitor(env, "./vid", video_callable=lambda episode_id: True,force=True)
    env.seed(config['seed'])
    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)
    agent = MDQNAgent(state_size=8, action_size=4, config=config)
    agent = MDDQNAgent(state_size=8, action_size=4, config=config)
    replay_buffer = ReplayBuffer((8, ), (1, ), int(args.buffer_size),  config['device'])
    # agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
    n_episodes = 1000
    eps = 1.0
    eps_end = 0.01
    eps_decay = 0.995
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)
    # eps = 1   # random policy
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
    pathname = dt_string + "seed_" + str(config['seed'])
    tensorboard_name = 'runs/' + pathname
    writer = SummaryWriter(tensorboard_name)
    t0 = time.time()
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        env_score = 0
        for t in range(args.max_episode_steps):    
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(replay_buffer)
            env_score += reward
            done_bool = 0 if t + 1 == args.max_episode_steps else float(done)
            replay_buffer.add(state, action, reward, next_state, done, done_bool)
            state = next_state
            if done:
                print("Episode {}  Reward {} steps {}".format(i_episode, env_score, t))
                break
        scores_window.append(env_score)       # save most recent score
        scores.append(env_score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f} epsilon {:.2f}'  .format(i_episode, np.mean(scores_window), eps), end="")
        mean_reward =  np.mean(scores_window)
        writer.add_scalar('env_reward', mean_reward, i_episode)
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f} Time: {}'.format(i_episode, np.mean(scores_window),  time_format(time.time()-t0)))
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--param', default="param.json", type=str)
    parser.add_argument('--lr', default=0, type=float)
    parser.add_argument('--fc1_units', default=256, type=int)
    parser.add_argument('--fc2_units', default=256, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--buffer_size', default=1e5, type=int)
    parser.add_argument('--max_episode_steps', default=1000, type=int) 
    parser.add_argument('--locexp', default="search_results", type=str) 
    arg = parser.parse_args()
    main(arg)

