import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


env = gym.make('LunarLander-v2')
env.seed(0)
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)




from sac_agent import Agent


def dqn(agent, writer, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, seed=0):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    env.seed(seed)
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    print("start training")
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state)
            if i_episode < 10:
                action = env.action_space.sample()

            next_state, reward, done, _ = env.step(action)
            clipped_reward = max(min(reward, 1.0), -1.0)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        mean = np.mean(scores_window)
        writer.add_scalar('Reward', score, i_episode)
        writer.add_scalar('Reward mean ', mean, i_episode)

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores



def eval(env, agent):
    score = 0
    for t in range(max_t):
        action = agent.act_greedy(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        score += reward
        if done:
            break
    print("Eval reward {}".format(score)



seed = 1
agent = Agent(state_size=8, action_size=4, seed=seed)
tensorboard_name = "DQN" + '/runs/' + "DQN-" + str(seed)
writer = SummaryWriter(tensorboard_name)
scores = dqn(agent, writer, seed=seed)

seed = 2
agent = Agent(state_size=8, action_size=4, seed=seed)
tensorboard_name = "DQN" + '/runs/' + "DQN-" + str(seed)
writer = SummaryWriter(tensorboard_name)
scores = dqn(agent, writer, seed=seed)


seed = 3
agent = Agent(state_size=8, action_size=4, seed=seed)
tensorboard_name = "DQN" + '/runs/' + "DQN-" + str(seed)
writer = SummaryWriter(tensorboard_name)
scores = dqn(agent, writer, seed=seed)


seed = 4
agent = Agent(state_size=8, action_size=4, seed=seed)
tensorboard_name = "DQN" + '/runs/' + "DQN-" + str(seed)
writer = SummaryWriter(tensorboard_name)
scores = dqn(agent, writer, seed=seed)


seed = 5
agent = Agent(state_size=8, action_size=4, seed=seed)
tensorboard_name = "DQN" + '/runs/' + "DQN-" + str(seed)
writer = SummaryWriter(tensorboard_name)
scores = dqn(agent, writer, seed=seed)


















