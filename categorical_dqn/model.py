import torch
import torch.nn as nn
import torch.nn.functional as F

class CategoricalDQN(nn.Module):
    def __init__(self, input_shape, num_actions, atoms=51):
        super(CategoricalDQN, self).__init__()

        self.input_shape = input_shape[0]
        self.num_actions = num_actions
        self.atoms = atoms
        print("input shape ", self.input_shape)
        self.fc1 = nn.Linear(self.input_shape, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, self.num_actions*self.atoms)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x.view(-1, self.num_actions, self.atoms), dim=2)

