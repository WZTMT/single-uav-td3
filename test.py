import numpy as np
import airsim

from model.td3 import TD3
from utils import save_results, plot_rewards
from torch.utils.tensorboard import SummaryWriter
from env.multirotor import Multirotor
from train import TD3Config, set_seed


def test(cfg, client, agent):
    print('Start Testing!')
    print(f'Env：{cfg.env_name}, Algorithm：{cfg.algo_name}, Device：{cfg.device}')
    rewards, ma_rewards = [], []
    writer = SummaryWriter('./test_image')
    success = 0
    for i_ep in range(cfg.test_eps):
        env = Multirotor(client, True)
        state = env.get_state()
        ep_reward = 0
        finish_step = 0
        final_distance = state[3] * env.max_distance
        for i_step in range(cfg.max_step):
            finish_step = finish_step + 1
            # action = (
            #         agent.choose_action(state) +
            #         np.random.normal(0, cfg.max_action * cfg.expl_noise, size=cfg.n_actions)
            # ).clip(-cfg.max_action, cfg.max_action)
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            state = next_state
            ep_reward += reward
            print('\rEpisode: {}\tStep: {}\tReward: {:.2f}\tDistance: {:.2f}'.format(i_ep + 1, i_step + 1, ep_reward, state[3] * env.max_distance), end="")
            final_distance = state[3] * env.max_distance
            if done:
                break
        print('\rEpisode: {}\tFinish step: {}\tReward: {:.2f}\tFinal distance: {:.2f}'.format(i_ep + 1, finish_step, ep_reward, final_distance))
        if final_distance <= 25.0:
            success += 1
        rewards.append(ep_reward)
        # 计算滑动窗口的reward
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        # writer.add_scalars(main_tag='test',
        #                    tag_scalar_dict={
        #                        'reward': ep_reward,
        #                        'ma_reward': ma_rewards[-1]
        #                    },
        #                    global_step=i_ep)
        if i_ep + 1 == cfg.test_eps:
            env.land()
    print('Finish Testing!')
    print('Average Reward: {}\tSuccess Rate: {}'.format(np.mean(rewards), success / cfg.test_eps))
    writer.close()
    return rewards, ma_rewards


# 全图成功率 45%
# 全图 0.41
# 全图 验证0.26 0.25 加噪声测试 不验证0.30
# 全图 0.18 加噪声测试 验证0.11 0.16 局部测试 0.85
if __name__ == "__main__":
    cfg = TD3Config()
    set_seed(cfg.seed)
    client = airsim.MultirotorClient()  # connect to the AirSim simulator
    agent = TD3(cfg)
    agent.load(cfg.model_path)
    rewards, ma_rewards = test(cfg, client, agent)
    # save_results(rewards, ma_rewards, tag="test", path=cfg.result_path)
    # plot_rewards(rewards, ma_rewards, cfg, tag="test")
