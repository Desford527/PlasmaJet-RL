
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PPO_AC_Def import Jet  # 导入环境类 Jet
from PPO_AC_Def import PPO  # 导入 PPO 算法类
from PPO_AC_Def import act_num  # 导入动作维度

# 关闭 TensorFlow 的 eager execution，使用静态计算图（提高性能）
tf.compat.v1.disable_eager_execution()

# 训练模式标志（1 代表训练，2 代表基线点追踪测试模式，3 代表抗干扰测试模式）
train = 1

# 定义 PPO 训练的超参数
ROLL_STEPS = 200  # 每次数据采集的回合数
STEPS = 100  # 每个回合的最大步数
EPISODES = 1000  # 训练的总回合数

all_ep_r = []  # 记录每个回合的奖励值
GAMMA = 0.9  # 折扣因子
Target_UPDATE_TIMES = 30  # 目标网络更新次数
ACTOR_UPDATE_TIMES = 30  # Actor 更新次数
CRITIC_UPDATE_TIMES = 50  # Critic 更新次数
# 神经网络参数
FC1_n = 256  # 第一层全连接层的神经元数
FC2_n = 126  # 第二层全连接层的神经元数
Critic_LR = 5e-5  # Critic 学习率
Actor_LR = 1e-5  # Actor 学习率

MODEL_NAME = 'model-3species-etop'  # 训练模型的名称

# 定义基线点追踪测试序列的参数
NUM = 100  # 每个阶段的点数，用于控制分辨率
AMPLITUDE = 1.2  # 正弦波的振幅
BASE = 3  # 正弦波的基值（中心位置）
FACTOR = 1  # 序列的整体缩放因子
CONSTANTS = [ 3.8, 1.5, 4.0, 5.1]  # 平稳阶段的常数值列表
# 生成线性变化阶段
linear_phase = np.linspace(0, 3, NUM)  # 从0.4线性增长到1.2，共NUM个点
# 生成正弦波变化阶段
sinusoidal_phase = BASE + AMPLITUDE * np.sin(np.linspace(0, 6 * np.pi, NUM))  # 基于sin函数生成波动变化
# 生成多个平稳阶段
constant_phases = [np.full(NUM, c) for c in CONSTANTS]  # 为每个常数值生成长度为NUM的数组
# 合并所有阶段形成完整的测试序列
test_seq = np.concatenate([linear_phase] + [sinusoidal_phase] + constant_phases)
# 应用缩放因子
test_seq = test_seq * FACTOR



def main():
    print(train)
    # 初始化环境和 PPO 代理
    env = Jet()
    agent = PPO(env.observation_space, env.action_space, Critic_LR, Actor_LR, Target_UPDATE_TIMES,
                ACTOR_UPDATE_TIMES, CRITIC_UPDATE_TIMES, FC1_n, FC2_n, MODEL_NAME , GAMMA)
    # 生成目标密度值
    target = np.random.uniform(0.2, 4.98, (ROLL_STEPS, 1))
    print(train)
    if train == 1:  # 训练模式
        for episode in range(EPISODES):
            agent.resetMemory()  # 清空经验存储
            ep_r = 0  # 记录当前回合的累计奖励

            for roll in range(ROLL_STEPS):
                env.target_density = np.array([target[roll, 0]])  # 设置环境目标密度
                state = env.reset()  # 重置环境
                for step in range(STEPS):
                    action = agent.Choose_Action(state)  # 选择动作
                    next_state, reward, done, info = env.step(action)  # 执行动作
                    agent.Store_Data(state, action, reward, next_state, done)  # 存储经验
                    state = next_state  # 更新状态
                    ep_r += reward  # 累计奖励

            agent.Train(next_state)  # 训练 PPO 代理
            agent.UpdateActorParameters(ep_r)  # 更新 Actor 网络参数

            all_ep_r.append(ep_r)  # 记录奖励值
            print('Ep: %i' % episode, "|Ep_r: %i" % ep_r, "action: ", action)

            if episode % 10 == 0 or episode == EPISODES - 1:
                outfile = pd.DataFrame(all_ep_r)
                outfile.to_csv(f'PPO_model/{MODEL_NAME}/{MODEL_NAME}-epoch.csv', sep=',', header=False, index=False)

        # 绘制训练奖励曲线
        plt.plot(all_ep_r)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.show()
    elif train == 2:
        """
                   基线点追踪测试模式：
                   - 在训练好的模型上运行测试数据 `test_seq`
                   - 观察模型如何跟随目标密度的变化
                   - 记录并绘制相应的输出数据
                   """

        agent.load_model()  # 加载训练好的 PPO 模型
        env.target_density = test_seq[0]  # 设置初始目标密度
        state = env.reset()  # 复位环境

        # 初始化数据记录数组
        out_v = np.array([])  # 记录模型输出误差
        action0 = np.array([])  # 记录动作 1（体积）
        action1 = np.array([])  # 记录动作 2（流速）
        action2 = np.array([])  # 记录动作 3（氧气浓度）
        act_list = np.array([])  # 记录所有动作

        # 遍历测试序列，依次将目标密度输入环境
        for T in test_seq:
            env.target_density = T  # 更新目标密度
            action = agent.Choose_Action(state)  # 选择当前状态下的动作
            next_state, reward, done, info = env.step(action)  # 执行动作，获取下一个状态及奖励

            # 记录误差（目标密度 - 当前密度）
            out_v = np.append(out_v, env.target_density - state[0][0])
            # 记录当前状态的部分变量
            action0 = np.append(action0, state[0][2])
            action1 = np.append(action1, state[0][4])
            action2 = np.append(action2, state[0][6])
            act_list = np.append(act_list, action)

            state = next_state  # 更新状态

        # 绘制测试结果
        plt.figure()
        plt.plot(action0, label='vol')
        plt.plot(action1, label='flow')
        plt.plot(action2, label='O2')
        plt.plot(out_v, label='out')
        plt.plot(test_seq, label='set')
        plt.legend()
        plt.xlabel("Step")
        plt.ylabel("Out_v")

        # 保存数据到 CSV 文件
        act_list = act_list.reshape(-1, act_num)
        outfile = pd.DataFrame(np.concatenate((test_seq.reshape(-1, 1), out_v.reshape(-1, 1), act_list), axis=1))
        outfile.to_csv(f'PPO_model/{MODEL_NAME}/{MODEL_NAME}-track.csv', sep=',')
        plt.show()


    elif train == 3:
            """
            抗干扰测试模式 ：
            - 测试模型在多个固定目标密度值下的跟踪能力
            - 在一定时间后加入水分扰动，观察系统稳定性
            """

            agent.load_model()  # 加载训练好的 PPO 模型
            target = [1, 1.5, 2,]  # 预设的固定目标密度值
            target_num = len(target)  # 目标密度值的个数

            # 遍历多个目标密度，进行测试
            for i in range(target_num):
                env.target_density = target[i]  # 设置目标密度
                state = env.reset()  # 重置环境

                # 初始化数据存储
                out_v, action0, action1, action2, act_list = np.array([]), np.array([]), np.array([]), np.array(
                    []), np.array([])

                for T in range(200):  # 运行 200 个时间步
                    action = agent.Choose_Action(state)  # 选择动作
                    next_state, reward, done, info = env.step(action, H2o=0.33 * (T // 50))  # 每 50 步改变水含量

                    # 记录输出误差和状态信息
                    out_v = np.append(out_v, env.target_density - state[0][0])
                    action0 = np.append(action0, state[0][2])
                    action1 = np.append(action1, state[0][4])
                    action2 = np.append(action2, state[0][6])
                    act_list = np.append(act_list, action)

                    state = next_state  # 更新状态

                # 绘制结果
                plt.figure()
                plt.plot(out_v, label='out')
                plt.plot(action0, label='vol')
                plt.plot(action1, label='flow')
                plt.plot(action2, label='O2')
                plt.legend()
                plt.xlabel("Step")
                plt.ylabel("Out_v")

            # 保存扰动测试数据
            act_list = act_list.reshape(-1, act_num)
            outfile = pd.DataFrame(np.concatenate((out_v.reshape(-1, 1), act_list), axis=1))
            outfile.to_csv(f'PPO_model/{MODEL_NAME}/{MODEL_NAME}-disturb.csv', sep=',', header=False, index=False)
            plt.show()

if __name__ == '__main__':
    main()
