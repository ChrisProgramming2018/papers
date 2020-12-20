import torch
import torch.nn as nn
import torch.nn.functional as F



class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, fc_1, fc_2, leak=0.01):
        self.leak = leak
        super(QNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.layer1 = nn.Linear(state_size + action_size, fc_1)
        self.layer2 = nn.Linear(fc_1, fc_2)
        self.layer3 = nn.Linear(fc_2, 1)
        self.reset_parameters()

    def reset_parameters(self):
        """ Initilaize the weights using He et al (2015) weights """
        torch.nn.init.kaiming_normal_(self.layer1.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.kaiming_normal_(self.layer2.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.uniform_(self.layer3.weight.data, -3e-4, 3e-4)

    def forward(self, state, action):
        xu = torch.cat([state, action], dim=1)
        x1 = F.relu(self.layer1(xu))
        x2 = F.relu(self.layer2(x1))
        x3 = self.layer3(x2)
        return x3




class Actor(nn.Module):
    def __init__(self, state_size, action_size, fc_1=64, fc_2=64, leak=0.01):
        super(Actor, self).__init__()
        self.leak = leak
        self.state_size = state_size
        self.action_size = action_size
        self.layer1 = nn.Linear(state_size, fc_1)
        self.layer2 = nn.Linear(fc_1, fc_2)
        self.layer3 = nn.Linear(fc_2, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        """ Initilaize the weights using He et al (2015) weights """
        torch.nn.init.kaiming_normal_(self.layer1.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.kaiming_normal_(self.layer2.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.uniform_(self.layer3.weight.data, -3e-3, 3e-3)

    def forward(self, state):
        x1 = F.relu(self.layer1(state))
        x2 = F.relu(self.layer2(x1))
        x3 = torch.tanh(self.layer3(x2))
        return x3

