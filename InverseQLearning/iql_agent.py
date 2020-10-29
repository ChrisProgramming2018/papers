import sys
import numpy as np
import random
from collections import namedtuple, deque
from models import QNetwork, RNetwork, PolicyNetwork, Classifier
import torch
import torch.nn  as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable


class Agent():
    def __init__(self, state_size, action_size, action_dim, config):
        self.state_size = state_size
        self.action_size = action_size
        self.action_dim = action_dim
        self.seed = 0
        self.device = 'cuda'
        self.batch_size = config["batch_size"]
        self.lr = 0.005
        self.gamma = 0.99
        self.q_shift_local = QNetwork(state_size, action_size, self.seed).to(self.device)
        self.q_shift_target = QNetwork(state_size, action_size, self.seed).to(self.device)
        self.Q_local = QNetwork(state_size, action_size, self.seed).to(self.device)
        self.Q_target = QNetwork(state_size, action_size, self.seed).to(self.device)
        self.R_local = RNetwork(state_size,action_size, self.seed).to(self.device)
        self.R_target = RNetwork(state_size, action_size, self.seed).to(self.device)
        self.policy = PolicyNetwork(state_size, action_size,self.seed).to(self.device)
        self.predicter = Classifier(state_size, action_dim, self.seed).to(self.device)
        #self.criterion = nn.CrossEntropyLoss()
        # optimizer
        self.optimizer_q_shift = optim.Adam(self.q_shift_local.parameters(), lr=self.lr)
        self.optimizer_q = optim.Adam(self.Q_local.parameters(), lr=self.lr)
        self.optimizer_r = optim.Adam(self.R_local.parameters(), lr=self.lr)
        self.optimizer_p = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.optimizer_pre = optim.Adam(self.predicter.parameters(), lr=self.lr)
        pathname = "lr {} batch_size {} seed {}".format(self.lr, self.batch_size, self.seed)
        tensorboard_name = str(config["locexp"]) + '/runs/' + pathname 
        self.writer = SummaryWriter(tensorboard_name)
        self.steps = 0
        self.ratio = 1. / action_dim
        self.all_actions = []
        for a in range(self.action_dim):
            action = torch.Tensor(1) * 0 +  a
            self.all_actions.append(action.to(self.device))

    def act(self, state):
        dis, action, log_probs, ent = self.policy.sample_action(torch.Tensor(state).unsqueeze(0))
        return dis, action, log_probs, ent

    def learn(self, memory):
        states, next_states, actions = memory.expert_policy(self.batch_size)
        # actions = actions[0]
        # print("states ",  states)
        self.state_action_frq(states, actions)
        self.get_action_prob(states, actions)
        self.compute_r_function(states, actions)
        return
        # compute difference between Q_shift and y_sh
        q_sh_value = self.q_shift_local(next_states, actions)
        y_sh = np.empty((self.batch_size,1), dtype=np.float32)
        for idx, s in enumerate(next_states):
            q = []
            for action in self.all_actions:
                q.append(Q_target(s.unsqueeze(0), action.unsqueeze(0)))
            q_max = max(q)
            np.copyto(y_sh[idx], q_max.detach().numpy())




        y_sh = torch.Tensor(y_sh)
        y_sh *= self.gamma 
        q_shift_loss = F.mse_loss(y_sh, q_shift_values)
        # Minimize the loss
        self.optimizer.zero_grad()
        q_shift_loss.backward()
        self.optimizer.step()

        #minimize MSE between pred Q and y = r'(s,a) + gama * max Q'(s',a)
        q_current = self.Q_local(states, actions)
        r_hat = self.R_target(states, actions)
        # use y_sh as target 
        y_q = r_hat + y_sh 
        
        q_loss = F.mse_loss(q_current, y_q)
        # Minimize the loss
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()

        
        
        
        
        #  get predicted reward
        r = self.R_local(states, actions)

    def state_action_frq(self, states, action):
        """ Train classifer to compute state action freq
        """
        self.steps +=1
        output = self.predicter(states)
        # create one hot encode y from actions
        y = action.type(torch.long)
        y = y.squeeze(1) 
        loss = nn.CrossEntropyLoss()(output, y)
        self.optimizer_pre.zero_grad()
        loss.backward()
        self.optimizer_pre.step()
        self.writer.add_scalar('Predict_loss', loss, self.steps)


    def get_action_prob(self, states, actions, dim=False):
        """

        """
        if dim:
            output = self.predicter(states)
            action_prob = output.gather(1, actions.type(torch.long))
            action_prob = torch.log(action_prob)
            return action_prob
        output = self.predicter(states)
        print("Output prob ", output)
        action_prob = output.gather(1, actions.type(torch.long))
        print("action prob ", action_prob)
        action_prob = torch.log(action_prob)
        print("action prob ", action_prob)
        return action_prob



    def compute_r_function(self, states, actions):
        """
        
        """
        actions = actions.type(torch.float)
        y = self.R_local(states, actions)
        y_shift = self.q_shift_target(states, actions)
        y_r_part1 = self.get_action_prob(states, actions) - y_shift
        print("ratio ", self.ratio)
        # sum all other actions
        y_r_part2 =  torch.empty((self.batch_size, 1), dtype=torch.float32)
        idx = 0
        for a, s in zip(actions, states):
            y_h = 0
            for b in self.all_actions:
                if torch.eq(a, b):
                    continue
                print("diff ac ", b)
                r_hat = self.R_target(s.unsqueeze(0), b.unsqueeze(0))
                n_b = self.get_action_prob(s.unsqueeze(0), b.unsqueeze(0), True) - self.q_shift_target(s.unsqueeze(0), b.unsqueeze(0))
                y_h += (r_hat - n_b)
            y_h = self.ratio * y_h
            y_r_part2[idx] = y_h
            idx += 1
        print("shape of r y ", y.shape)
        print("y r part 1 ", y_r_part1.shape)
        print("y r part 2 ", y_r_part2.shape)











