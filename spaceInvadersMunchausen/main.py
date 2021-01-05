import os
import sys
import random
import torch
import numpy as np
from mdqn_agent import MDQNAgent
from mddqn_agent import MDDQNAgent
from torch.utils.tensorboard import SummaryWriter
import argparse
import json
import time
import gym
from datetime import datetime
from collections import namedtuple, deque

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
    env = gym.make(config["env_name"])
    config["seed"] = args.seed
    path = config["locexp"]
    if not os.path.exists(path):
        os.makedirs(path)
    print('State shape: ', env.observation_space.shape)
    action_space = env.action_space.n
    print('Number of actions: ', env.action_space.n)
    state_size = 200
    agent = MDDQNAgent(state_size=state_size, action_size=action_space, config=config)
    agent.train()


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

