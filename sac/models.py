import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, fc_1, fc_2):
        super(QNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.layer1 = nn.Linear(state_size + action_size, fc_1)
        self.layer2 = nn.Linear(fc_1, fc_2)
        self.layer3 = nn.Linear(fc_2, 1)
        self.layer4 = nn.Linear(state_size + action_size, fc_1)
        self.layer5 = nn.Linear(fc_1, fc_2)
        self.layer6 = nn.Linear(fc_2, 1)

    def forward(self, state, action):
        xu = torch.cat([state, action], dim=1)
        x1 = F.relu(self.layer1(xu))
        x2 = F.relu(self.layer2(x1))
        x3 = self.layer3(x2)

        x4 = F.relu(self.layer4(xu))
        x5 = F.relu(self.layer5(x4))
        x6 = self.layer6(x5)
        return x3, x6
    
    def Q1(self, state, action):
        xu = torch.cat([state, action], dim=1)
        x1 = F.relu(self.layer1(xu))
        x2 = F.relu(self.layer2(x1))
        x3 = self.layer3(x2)
        return x3




class SACActor(nn.Module):
    def __init__(self, state_size, action_size, log_sig_min=-20, log_sig_max=2, fc_1=64, fc_2=64, action_space=None):
        super(SACActor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.layer1 = nn.Linear(state_size, fc_1)
        self.layer2 = nn.Linear(fc_1, fc_2)
        self.mean_layer = nn.Linear(fc_2, action_size)
        self.log_std_linear_layer = nn.Linear(fc_2, action_size)
        self.log_sig_min = log_sig_min
        self.log_sig_max = log_sig_max
         # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)


    def forward(self, state):
        x1 = F.relu(self.layer1(state))
        x2 = F.relu(self.layer2(x1))
        mean = self.mean_layer(x2)
        log_std = self.log_std_linear_layer(x2)
        log_std = torch.clamp(log_std, min=self.log_sig_min, max=self.log_sig_max)
        return mean, log_std


    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample() # (mean + std * Normal(0,1))
        y_t = torch.tanh(x_t)
        # Get values in bound
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
