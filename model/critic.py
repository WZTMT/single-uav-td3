import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, n_states, n_actions, init_w=3e-3):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(n_states + n_actions, 256)
        self.l2 = nn.Linear(256, 256)  # 最近一次训练没有第二层和第三层
        self.l3 = nn.Linear(256, 256)  # 最近一次训练没有第二层和第三层
        self.l4 = nn.Linear(256, 256)
        self.l5 = nn.Linear(256, 1)

        nn.init.uniform_(self.l5.weight.detach(), a=-init_w, b=init_w)
        nn.init.uniform_(self.l5.bias.detach(), a=-init_w, b=init_w)

        # Q2 architecture
        self.l6 = nn.Linear(n_states + n_actions, 256)
        self.l7 = nn.Linear(256, 256)  # 最近一次训练没有第七层和第八层
        self.l8 = nn.Linear(256, 256)  # 最近一次训练没有第七层和第八层
        self.l9 = nn.Linear(256, 256)
        self.l10 = nn.Linear(256, 1)

        nn.init.uniform_(self.l10.weight.detach(), a=-init_w, b=init_w)
        nn.init.uniform_(self.l10.bias.detach(), a=-init_w, b=init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(x))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        q1 = F.relu(self.l4(q1))
        q1 = torch.tanh(self.l5(q1))

        q2 = F.relu(self.l6(x))
        q2 = F.relu(self.l7(q2))
        q2 = F.relu(self.l8(q2))
        q2 = F.relu(self.l9(q2))
        q2 = torch.tanh(self.l10(q2))
        return q1, q2

    def q1(self, state, action):
        x = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(x))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        q1 = F.relu(self.l4(q1))
        q1 = torch.tanh(self.l5(q1))
        return q1
