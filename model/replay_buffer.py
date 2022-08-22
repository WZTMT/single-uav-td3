import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, n_states, n_actions, max_size=int(2**17)):
        self.max_size = max_size
        self.position = 0  # 当前要存储在哪一位
        self.size = 0  # 当前长度
        self.state = np.zeros((max_size, n_states))
        self.action = np.zeros((max_size, n_actions))
        self.next_state = np.zeros((max_size, n_states))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def push(self, state, action, next_state, reward, done):
        self.state[self.position] = state
        self.action[self.position] = action
        self.next_state[self.position] = next_state
        self.reward[self.position] = reward
        self.not_done[self.position] = 1. - done
        self.position = (self.position + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    # 返回5项二维tensor，每项的一维长度都是batch_size
    def sample(self, batch_size):
        index = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[index]).to(self.device),
            torch.FloatTensor(self.action[index]).to(self.device),
            torch.FloatTensor(self.next_state[index]).to(self.device),
            torch.FloatTensor(self.reward[index]).to(self.device),
            torch.FloatTensor(self.not_done[index]).to(self.device)
        )

    def __len__(self):
        return self.size


if __name__ == '__main__':
    state = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    action = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    ind = np.random.randint(0, 2**17, size=25)
