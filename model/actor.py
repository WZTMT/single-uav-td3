import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """
    input_dim: 输入维度，这里等于n_states
    output_dim: 输出维度，这里等于n_actions
    max_action: action的最大值
    """
    def __init__(self, n_states, n_actions, max_action, init_w=3e-3):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(n_states, 256)
        # self.l2 = nn.Linear(256, 256)
        # self.l3 = nn.Linear(256, 256)
        self.l4 = nn.Linear(256, 256)
        self.l5 = nn.Linear(256, n_actions)
        self.max_action = max_action

        nn.init.uniform_(self.l5.weight.detach(), a=-init_w, b=init_w)
        nn.init.uniform_(self.l5.bias.detach(), a=-init_w, b=init_w)

    def forward(self, state):
        x = F.relu(self.l1(state))
        # x = F.relu(self.l2(x))
        # x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        action = torch.tanh(self.l5(x))  # torch.tanh与F.tanh没有区别
        return self.max_action * action
