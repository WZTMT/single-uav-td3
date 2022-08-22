import torch
import copy
import torch.nn.functional as F
import numpy as np

from model.actor import Actor
from model.critic import Critic
from model.replay_buffer import ReplayBuffer


class TD3(object):
    def __init__(self, cfg):
        self.max_action = cfg.max_action
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.policy_noise = cfg.policy_noise  # 策略平滑正则化：加在目标策略上的噪声，用于防止critic过拟合
        self.noise_clip = cfg.noise_clip  # 噪声的最大值
        self.policy_freq = cfg.policy_freq  # 策略网络延迟更新，更新频率
        self.batch_size = cfg.batch_size
        self.device = cfg.device
        self.n_actions = cfg.n_actions
        self.total_iteration = 0  # 模型的更新次数

        self.actor = Actor(cfg.n_states, cfg.n_actions, self.max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)  # 将一个网络赋值给另一个网络，且不相互影响
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(cfg.n_states, cfg.n_actions).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.memory = ReplayBuffer(cfg.n_states, cfg.n_actions, max_size=cfg.memory_capacity)

    '''
    根据一次的状态选出一次动作
    '''
    def choose_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state)
        action = action.cpu().data.numpy().flatten()  # flatten()将高维数组展成一维向量
        return action

    def update(self):
        if len(self.memory) < self.batch_size:  # 当 memory 中不满足一个批量时，不更新策略
            return

        self.total_iteration += 1

        # Sample replay buffer 取一个batch_size的数据
        state, action, next_state, reward, not_done = self.memory.sample(self.batch_size)

        # 所有在该模块下计算出的tensor的required_grad都为false，都不会被求导
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            # torch.randn_like()返回一个均值为0，方差为1的高斯分布的tensor
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            # noise = torch.FloatTensor(np.random.normal(0, self.max_action * self.policy_noise, size=self.n_actions).clip(-self.noise_clip, self.noise_clip)).to(self.device)

            next_action = self.actor_target(next_state)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + not_done * self.gamma * target_q  # 贝尔曼方程形式

        # Get current Q estimates
        current_q1, current_q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_iteration % self.policy_freq == 0:

            # Compute actor loss
            actor_loss = -self.critic.q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, path):
        torch.save(self.critic.state_dict(), path + "td3_critic")
        torch.save(self.critic_optimizer.state_dict(), path + "td3_critic_optimizer")

        torch.save(self.actor.state_dict(), path + "td3_actor")
        torch.save(self.actor_optimizer.state_dict(), path + "td3_actor_optimizer")

    def load(self, path):
        self.critic.load_state_dict(torch.load(path + "td3_critic"))
        self.critic_optimizer.load_state_dict(torch.load(path + "td3_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(path + "td3_actor"))
        self.actor_optimizer.load_state_dict(torch.load(path + "td3_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)


if __name__ == '__main__':
    action = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
    noise = torch.randn_like(action).clamp(-.5, .5)
    print(noise)
