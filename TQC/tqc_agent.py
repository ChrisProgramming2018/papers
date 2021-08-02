import os
import time
import torch
import copy
import numpy as np
from cutin_policy.tqc_model import Actor, Critic,  quantile_huber_loss_f
import torchvision.transforms.functional as TF
from cutin_policy.replay_buffer import ReplayBuffer
# Building the whole Training Process into a class
import json
from cutin_policy.utils import time_format
import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from cutin_envs.sceneries import first_scenario, traffic_lights_scenario
from cutin_policy.helper import  write_into_file
import matplotlib.pyplot as plt


class TQC(object):
    def __init__(self, state_dim, action_dim, config):
        self.actor = Actor(state_dim, action_dim, config).to(config["device"])
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), config["lr_actor"])        
        self.critic = Critic(state_dim, action_dim, config).to(config["device"])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), config["lr_critic"])
        self.target_critic = Critic(state_dim, action_dim, config).to(config["device"])
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.device = config["device"]
        self.batch_size = int(config["batch_size"])
        self.discount = config["discount"]
        self.tau = config["tau"]
        self.device = config["device"]
        self.eval = 50
        self.write_tensorboard = False
        self.top_quantiles_to_drop = config["top_quantiles_to_drop_per_net"] * config["n_nets"]
        self.target_entropy = config["target_entropy"]
        self.quantiles_total = self.critic.n_quantiles * self.critic.n_nets
        self.log_alpha = torch.zeros((1,), requires_grad=True, device=config["device"])
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config["lr_alpha"])
        self.total_it = 0
        self.step = 0
        self.seed = config["seed"]
        self.episodes = 100000
        self.locexp = str(config["locexp"])
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
        pathname = dt_string + "seed_" + str(config["seed"])
        self.memory = ReplayBuffer((state_dim,), (1,), int(config["buffer_size"]), self.seed, config["device"])
        tensorboard_name = str(config["locexp"]) + "/runs/" + pathname
        self.vid_path = str(config["locexp"]) + "/vid"
        self.writer = SummaryWriter(tensorboard_name)
        self.env = traffic_lights_scenario(self.writer)
        self.steps = 0
        self.time_run = now.strftime("%d_%m_%Y_%H:%M:%S")

    def update(self, iterations):
        self.step += 1
        if self.step % 1000 == 0:
            self.write_tensorboard = 1 - self.write_tensorboard
        for it in range(iterations):
            # Step 4: We sample a batch of transitions (s, s’, a, r) from the memoy
            #sys.stdout = open(os.devnull, "w")
            state, action, reward, next_state, done = self.memory.sample(self.batch_size)

            #sys.stdout = sys.__stdout__
            # for augment 1

            alpha = torch.exp(self.log_alpha)
            with torch.no_grad(): 
                # Step 5: Get policy action
                new_next_action, next_log_pi =  self.actor(next_state)
                # compute quantile at next state
                next_z = self.target_critic(next_state, new_next_action)
                sorted_z, _ = torch.sort(next_z.reshape(self.batch_size, -1))
                sorted_z_part = sorted_z[:,:self.quantiles_total - self.top_quantiles_to_drop]
                target = reward + done * self.discount * (sorted_z_part - alpha * next_log_pi)
            #---update critic
            cur_z = self.critic(state, action)
            critic_loss = quantile_huber_loss_f(cur_z, target, self.device)
            self.writer.add_scalar('Critic_loss', critic_loss, self.step)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
        
            for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            #---Update policy and alpha
            new_action, log_pi = self.actor(state) # detached
            alpha_loss = -self.log_alpha * (log_pi + self.target_entropy).detach().mean()
            actor_loss = (alpha * log_pi - self.critic(state, new_action).mean(2).mean(1, keepdim=True)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.writer.add_scalar('Actor_loss', actor_loss, self.step)
            self.actor_optimizer.step()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.writer.add_scalar('Alpha_loss', actor_loss, self.step)
            self.alpha_optimizer.step()
            self.total_it +=1
    
    def select_action(self, obs):
        state = torch.as_tensor(obs, device=self.device, dtype=torch.float)
        #print(state.shape)
        if state.shape[0] != self.batch_size:
            state = state.unsqueeze(0)
        #print(state.shape)
        return self.actor.select_action(state)

    def save(self, filename):
        mkdir("", filename)
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
                
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

        torch.save(self.log_alpha, filename + "_alpha")
        torch.save(self.alpha_optimizer.state_dict(), filename + "_alpha_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
        self.log_alpha = torch.load(filename + "_alpha")
        self.alpha_optimizer.load_state_dict(torch.load(filename + "_alpha_optimizer"))

    def train_agent(self):
        scores_window = deque(maxlen=100)
        step_window = deque(maxlen=100)
        t = 0
        t0 = time.time()
        self.eval_policy(0)
        for i_epiosde in range(0, self.episodes):
            self.steps +=1
            episode_reward = 0
            state = self.env.reset()
            text = 'reset'
            text += 'state_0 : {:.2f}\n'.format(state[0])
            text += 'state_1 : {:.2f}\n'.format(state[1])
            text += 'state_2 : {:.2f}\n'.format(state[2])
            # write_into_file('log', text)
            average_action = []
            cutin = True
            epi_step = 0
            while True:
                t += 1

                if t < config["start_timesteps"]:
                    action = np.random.uniform(-10, 10)
                else:  # After 10000 timesteps, we switch to the model
                    #print(state.dtype)
                    action = self.select_action(state)[0] * 10
                # action = 20
                #print("action", action)
                average_action.append(action)
                next_state, reward, done, info = self.env.step(action)
                """
                text = 'time step {}\n'.format(epi_step)
                text += 'state_0 : {:.2f}\n'.format(state[0])
                text += 'state_1 : {:.2f}\n'.format(state[1])
                text += 'state_2 : {:.2f}\n'.format(state[2])
                #write_into_file('log', text)
                """
                # self.env.live_render()
                # self.steps += 1
                text = "state {} step {}".format(next_state, t)
                write_into_file('log', text)
                text = "traffic ligth {}".format(self.env.traffic_lights_state)
                write_into_file('log', text)
                if next_state is None:
                    break
                # print(info['cutin'])
                """
                if info['cutin'][0] == 1 and cutin:
                    cutin = False
                    cut_time_step = epi_step

                """
                episode_reward += reward
                if i_epiosde > 10:
                    self.update(1)
                self.memory.add(state, action, reward, next_state, done, done)
                state = next_state
                if done:
                    break
            if i_epiosde % self.eval == 0:
                self.save(self.locexp + "/models/model-{}".format(i_epiosde))
                self.eval_policy(i_epiosde)

            step_window.append(epi_step)
            scores_window.append(episode_reward)
            average_action = np.mean(np.array(average_action))
            min_action = np.min(np.array(average_action))

            ave_reward = np.mean(scores_window)
            mean_steps = np.mean(np.array(step_window))
            cut_time_step = 0
            print(
                "Epi {} Steps {} Re {:.2f} re av. {:.2f} cutins t. {} act {:.2f} Time {}".format(
                    i_epiosde,
                    t,
                    episode_reward,
                    np.mean(scores_window),
                    cut_time_step,
                    average_action,
                    time_format(time.time() - t0),
                )
            )
            self.writer.add_scalar("Aver_reward", ave_reward, self.steps)
            self.writer.add_scalar("steps_mean", mean_steps, self.steps)
            self.writer.add_scalar("min_action", min_action, self.steps)
            self.writer.add_scalar("episode_actions", average_action, self.steps)

    def eval_policy(self, eval_after_episode,  episodes=1):
        """  """
        eval_reward = []
        cutin_step_list = []
        average_action_list = []
        path = self.time_run
        # path2 = "images-{}".format(eval_after_episode)
        # path = os.path.join(path1, path2)

        try:
            os.makedirs(path)
        except FileExistsError:
            print("path {} already exist".format(path))

        for i_episode in range(1, episodes + 1):
            episode_reward = []
            cutin = True
            epi_step = 0
            cut_time_step = -1
            eps = 0  # choose always greedy action
            cutin_list = []
            average_action = []
            ego_state = []
            target_state = []
            ego_v = []
            target_v = []
            cutin_prob = []
            state = self.env.reset()
            text = "-----------------episode-{}--------------------".format(eval_after_episode)
            write_into_file('log', text)
            while True:
                epi_step += 1
                action = self.select_action(state)[0] * 10
                average_action.append(action)  # remove of set
                state, reward, done, info = self.env.step(action)
                episode_reward.append(reward)
                ego_state.append(state[0])
                ego_v.append(state[1])
                if state is None:
                    break
                """
                cutin_prob.append(state[6])
                ego_v.append(self.env.ego_state[1])
                target_v.append(state[4])
                ego_state.append(self.env.ego_state[0])
                target_state.append(state[3])
                text = 'time step {}\n'.format(epi_step)
                text += 'state_0 : {:.2f}\n'.format(state[0])
                text += 'state_1 : {:.2f}\n'.format(state[1])
                text += 'state_2 : {:.2f}\n'.format(state[2])
                text += 'state_3 : {:.2f}\n'.format(state[3])
                text += 'state_4 : {:.2f}\n'.format(state[4])
                text += 'state_5 : {:.2f}\n'.format(state[6])
                text += '------------------------------------------'
                """
                # write_into_file('log', text)
                """
                if info['cutin'][0] == 1 and cutin:
                    cutin = False
                    text = '----------------cutin--------------'
                    write_into_file('log', text)
                    cut_time_step = epi_step
                    cutin_list.append(1)
                else:
                    cutin_list.append(0)
                episode_reward += reward
                """
                if done:
                    text = 'velocity end {}'.format(state[1])
                    write_into_file('log', text)
                    break
            plt.clf()
            x = [i for i in range(len(average_action))]
            plt.plot(x, average_action)
            plt.savefig(path + '/actions_after_{}_steps.png'.format(eval_after_episode))
            reward = np.sum(episode_reward)

            """ 

            x = [i for i in range(len(cutin_list))]
            plt.plot(x, cutin_list)
            plt.savefig(path + '/cutin_after_{}_steps.png'.format(i_episode))
            """
            plt.clf()
            x = [i for i in range(len(ego_v))]
            plt.plot(x, ego_v, 'g', label='ego_v')
            plt.plot(x, episode_reward, 'r', label='reward')
            plt.plot(x, self.env.traffic_lights_state[:len(ego_v)], 'b', linestyle='dashed', label='trafficlight_state')
            # plt.plot(x, target_state, 'k', linestyle='dashed', label='target_state')
            plt.legend()
            plt.savefig(path + '/eval_steps_{}_reward{:.2f}.png'.format(eval_after_episode, reward))
            average_action_list.append(np.mean(np.array(average_action)))
            eval_reward.append(episode_reward)
            cutin_step_list.append(cut_time_step)
        if len(eval_reward) > 0:
            eval_reward = np.mean(np.array(eval_reward))
        else:
            eval_reward = -1

        if len(average_action_list) > 0:
            print(average_action_list)
            mean_action = np.mean(np.array(average_action_list))
        else:
            mean_action = -1
        self.writer.add_scalar("Eval_reward", eval_reward, self.steps)
        self.writer.add_scalar("Eval_ave_action", mean_action, self.steps)
            
def mkdir(base, name):
    """
    Creates a direction if its not exist
    Args:
       param1(string): base first part of pathname
       param2(string): name second part of pathname
    Return: pathname 
    """
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--param", default="param.json", type=str)
    args = parser.parse_args()
    with open(args.param, "r") as f:
        config = json.load(f)
    print(config)

    config["target_entropy"] = -np.prod(1)
    agent = TQC(5, 1, config=config)
    agent.train_agent()
