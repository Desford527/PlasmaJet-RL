import gym  # OpenAI Gym 库，用于创建和管理强化学习环境
from gym.spaces import Box  # Box 类用于定义连续的动作空间和状态空间
import tensorflow as tf  # TensorFlow 1.x 版本，用于构建深度学习模型
import pandas as pd  # Pandas 库，用于数据处理和表格操作
from DNN_Def import de_norm  # 从 DNN_Def 模块中导入 de_norm 函数，用于数据反归一化
import numpy as np  # NumPy 库，用于高效的数值计算
import os  # os 库用于文件和目录操作

import ast  # ast（抽象语法树）库，用于解析字符串形式的 Python 表达式
import csv  # csv 库，用于处理 CSV 文件的读写

# 读取CSV文件，加载 YMax_Arr, YMin_Arr, XMax_Arr, XMin_Arr
YMax_Arr = []  # 存储 Y 轴的最大值
YMin_Arr = []  # 存储 Y 轴的最小值
XMax_Arr = []  # 存储 X 轴的最大值（数组形式）
XMin_Arr = []  # 存储 X 轴的最小值（数组形式）

max_min_file_path = "max_min_values.csv"  # 归一化最大最小值数据存储的 CSV 文件路径

# 读取 CSV 文件并解析数据
with open(max_min_file_path, mode='r') as file:
    reader = csv.reader(file)  # 创建 CSV 读取器
    next(reader)  # 跳过第一行（标题行）
    for row in reader:
        YMax_Arr.append(float(row[0]))  # 解析并存储 Y 轴最大值
        YMin_Arr.append(float(row[1]))  # 解析并存储 Y 轴最小值
        XMax_Arr.append(np.array(ast.literal_eval(row[2])))  # 解析字符串形式的数组并转换为 NumPy 数组
        XMin_Arr.append(np.array(ast.literal_eval(row[3])))

# 定义强化学习相关的参数
act_num = 3  # 代表动作空间的维度（动作数量）
hist_num = 2  # 代表历史动作记录的长度

# 定义策略更新方法，选择 PPO 方法中的裁剪方法（clip）
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),  # KL 散度惩罚方法
    dict(name='clip', epsilon=0.3)  # 裁剪策略目标（Clipped Surrogate Objective），此方法效果更佳
][1]  # 选取 'clip' 方法进行优化

# 定义 PPO（近端策略优化）强化学习算法类
class PPO:
    def __init__(self, observation_space, action_space, critic_lr, actor_lr, target_episodes,
                 actor_episodes, critic_episodes, fc1_size, fc2_size, name, GAMMA):
        """
        PPO 代理初始化
        :param observation_space: 状态空间
        :param action_space: 动作空间
        :param critic_lr: Critic（价值网络）的学习率
        :param actor_lr: Actor（策略网络）的学习率
        :param target_episodes: 目标网络更新的频率
        :param actor_episodes: 策略网络更新的频率
        :param critic_episodes: 价值网络更新的频率
        :param fc1_size: 第一隐藏层神经元个数
        :param fc2_size: 第二隐藏层神经元个数
        :param name: 代理名称
        :param GAMMA: 折扣因子
        """
        self.state_dim = observation_space.shape[0]  # 获取状态空间的维度
        self.action_dim = action_space.shape[0]  # 获取动作空间的维度
        self.state_input = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, self.state_dim], 'state')  # 状态输入占位符
        # 初始化超参数
        self.old_r = -4000000  # 记录历史最优奖励
        self.Critic_LR = critic_lr  # 价值网络学习率
        self.Actor_LR = actor_lr  # 策略网络学习率
        self.Target_UPDATE_TIMES = target_episodes  # 目标网络更新间隔
        self.ACTOR_UPDATE_TIMES = actor_episodes  # 策略网络更新间隔
        self.CRITIC_UPDATE_TIMES = critic_episodes  # 价值网络更新间隔
        self.fc1_size = fc1_size  # 第一层全连接网络大小
        self.fc2_size = fc2_size  # 第二层全连接网络大小
        self.Create_Critic()        # 创建 Critic（价值网络）
        self.Create_Actor_with_two_network()        # 创建 Actor（策略网络）

        # 初始化 TensorFlow 会话
        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        # 保存和加载模型
        self.saver = tf.compat.v1.train.Saver()
        self.model_name = name

        self.GAMMA = GAMMA  # 折扣因子
        # 判断是否已经存在该模型的目录
        if not os.path.exists('PPO_model/' + self.model_name): # 若目录不存在，则创建
            os.makedirs('PPO_model/' + self.model_name)
            print(self.model_name + ' 创建成功')
        else:
            print(self.model_name + ' 目录已存在')

    # the critic network give the value of state
    def Create_Critic(self):
        """
        创建 Critic 网络（价值网络），用于评估当前状态的价值
        """
        # 第一层神经网络权重和偏置
        W1 = self.weight_variable([self.state_dim, self.fc1_size])
        b1 = self.bias_variable([self.fc1_size])
        # 第二层神经网络权重和偏置
        W2 = self.weight_variable([self.fc1_size, self.fc2_size])
        b2 = self.bias_variable([self.fc2_size])
        # 第三层神经网络权重和偏置（输出层）
        W3 = self.weight_variable([self.fc2_size, self.action_dim])
        b3 = self.bias_variable([self.action_dim])
        h_layer_one = tf.compat.v1.nn.relu(tf.compat.v1.matmul(self.state_input, W1) + b1) # 第一隐藏层，使用 ReLU 激活函数
        h_layer_two = tf.compat.v1.nn.relu(tf.compat.v1.matmul(h_layer_one, W2) + b2)        # 第二隐藏层，使用 ReLU 激活函数
        self.v = tf.compat.v1.matmul(h_layer_two, W3) + b3        # 输出层，计算状态值

        # 计算目标值与估计值的均方误差损失
        self.tfdc_r = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, 1], 'discounted_r')
        self.advantage = self.tfdc_r - self.v  # 计算优势函数
        self.closs = tf.compat.v1.reduce_mean(tf.compat.v1.square(self.advantage))  # 均方误差损失
        self.ctrain_op = tf.compat.v1.train.AdamOptimizer(self.Critic_LR).minimize(self.closs)  # Adam 优化器
        return

    # the actor network that give the action
    def Create_Actor_with_two_network(self):
        """
        Actor 网络用于生成动作策略 π(a|s)
        采用两个策略网络：
        - pi: 训练中的策略网络
        - oldpi: 旧策略网络（用于重要性采样）
        """
        pi, pi_params = self.build_actor_net('pi', trainable=True)  # 训练策略网络
        oldpi, oldpi_params = self.build_actor_net('oldpi', trainable=False)  # 旧策略网络（冻结参数）

        # 采样动作
        with tf.compat.v1.variable_scope('sample_action'):
            self.sample_from_oldpi = tf.squeeze(oldpi.sample(1), axis=0)  # 从旧策略网络采样动作
        # update the oldpi by coping the parameters from pi
        # 复制 pi 网络参数到 oldpi（用于重要性采样）
        with tf.compat.v1.variable_scope('update_oldpi'):
            self.update_oldpi_from_pi = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        # 定义 Actor 网络的训练目标
        self.tfa = tf.compat.v1.placeholder(tf.float32, [None, self.action_dim], 'action')  # 动作占位符
        self.tfadv = tf.compat.v1.placeholder(tf.float32, [None, self.action_dim], 'advantage')  # 优势函数 A(s, a) 占位符

        # 定义策略损失函数
        with tf.compat.v1.variable_scope('loss'):
            with tf.compat.v1.variable_scope('surrogate'):
                ratio = pi.prob(self.tfa) / (oldpi.prob(self.tfa) + 1e-5)  # 计算策略比率（重要性采样）
                surr = ratio * self.tfadv  # 计算策略目标函数
            # PPO 剪裁损失函数，确保策略更新不会过大
            self.aloss = -tf.compat.v1.reduce_mean(tf.minimum(
                surr,
                tf.clip_by_value(ratio, 1. - METHOD['epsilon'], 1. + METHOD['epsilon']) * self.tfadv
            ))
            # 使用 Adam 优化器最小化 Actor 损失
        with tf.compat.v1.variable_scope('atrain'):
            self.atrain_op = tf.compat.v1.train.AdamOptimizer(self.Actor_LR).minimize(self.aloss)
        return

    def build_actor_net(self, name, trainable):
        """
        构建 Actor 策略网络
        该网络用于生成动作分布 π(a|s)
        """
        with tf.compat.v1.variable_scope(name):
            l1 = tf.compat.v1.layers.dense(self.state_input, self.fc1_size, tf.nn.relu, trainable=trainable)
            l2 = tf.compat.v1.layers.dense(l1, self.fc2_size, tf.nn.relu, trainable=trainable)
            mu = tf.compat.v1.layers.dense(l2, self.action_dim, tf.nn.relu, trainable=trainable)
            sigma = tf.compat.v1.layers.dense(l2, self.action_dim, tf.nn.softplus, trainable=trainable)  # 标准差
            norm_dist = tf.compat.v1.distributions.Normal(loc=mu, scale=sigma)  # 生成正态分布
        params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=name)  # 获取网络参数
        return norm_dist, params  # 返回动作分布及参数

    # output the action with state, the output is from oldpi
    def Choose_Action(self, s):
        # s = s[np.newaxis, :]
        a = self.sess.run(self.sample_from_oldpi, {self.state_input: s})[0]
        return np.clip(a, 0, 1)

    # reset the memory in every episode
    def resetMemory(self):
        self.buffer_s, self.buffer_a, self.buffer_r = [], [], []

    # store the data of every steps
    def Store_Data(self, state, action, reward, next_state, done):
        self.buffer_s.append(state)
        self.buffer_a.append(action)
        self.buffer_r.append(reward)

    # get the state value from critic
    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.state_input: s})[0, 0]

    # the function that update the actor and critic
    def update(self, s, a, r):
        adv = self.sess.run(self.advantage, {self.state_input: s, self.tfdc_r: r})
        # update critic
        [self.sess.run(self.ctrain_op, {self.state_input: s, self.tfdc_r: r})
         for _ in range(self.CRITIC_UPDATE_TIMES)]
        # update actor
        [self.sess.run(self.atrain_op, {self.state_input: s, self.tfa: a, self.tfadv: adv})
         for _ in range(self.ACTOR_UPDATE_TIMES)]

    # the train function that update the network
    def Train(self, next_state):
        # calculate the discount reward
        for i in range(self.Target_UPDATE_TIMES):
            v_s_ = self.get_v(next_state)
            discounted_r = []
            for r in self.buffer_r[::-1]:  # 将reward倒序求解前面的v_s_
                v_s_ = r + self.GAMMA * v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()
            # discounted_r = discounted_r/(np.std(discounted_r) + 1e-5)
            bs, ba, br = np.vstack(self.buffer_s), np.vstack(self.buffer_a), np.array(discounted_r)[:, np.newaxis]
            self.update(bs, ba, br)

    # ths function to copy the pi's parameters to oldpi
    def UpdateActorParameters(self, r):
        self.sess.run(self.update_oldpi_from_pi)
        if r > self.old_r:
            self.saver.save(self.sess, save_path='PPO_model/' + self.model_name + '/' + self.model_name)
            self.old_r = r

    # the function that gives the weight initial value
    def weight_variable(self, shape):
        initial = tf.compat.v1.truncated_normal(shape)
        return tf.compat.v1.Variable(initial)

    # the function that gives the bias initial value
    def bias_variable(self, shape):
        initial = tf.compat.v1.constant(0.01, shape=shape)
        return tf.compat.v1.Variable(initial)

    def load_model(self):
        self.saver.restore(self.sess, 'PPO_model/' + self.model_name + '/' + self.model_name)


class Jet(object):
    def __init__(self):
        """
        初始化 Jet 类实例
        """
        # 初始化目标密度和输出密度
        self.target_density = np.array([[0]], dtype='float32')  # 目标密度，初始化为0
        self.output_density = np.array([[0] * hist_num], dtype='float32')  # 输出密度，初始化为0


        # 初始化差距
        self.distance = self.target_density - self.output_density  # 目标与当前输出的差异


        # 初始化动作，空的初始化为全0
        self.action = np.array([[0] * hist_num * act_num], dtype='float32')  # 动作初始化为零
        self.state = np.concatenate((self.distance, self.action), axis=1)  # 状态由密度差和动作组成

        # 加载系统模型，使用之前训练好的DNN模型
        self.SysModel_NO = tf.keras.models.load_model('DNN_model/DNN_Model_NO')  # 加载DNN模型，预测NO密度
        self.SysModel_O = tf.keras.models.load_model('DNN_model/DNN_Model_O')  # 加载DNN模型，预测O密度
        self.SysModel_OH = tf.keras.models.load_model('DNN_model/DNN_Model_OH')  # 加载DNN模型，预测OH密度

        # 初始化动作空间的上下限
        action_low = np.array([0] * act_num)  # 动作空间下限，默认为0
        action_high = np.array([1] * act_num)  # 动作空间上限，默认为1
        self.action_space = Box(low=action_low, high=action_high, dtype=np.float32)  # 动作空间：Box类型，值范围为[0, 1]

        # 初始化状态空间的上下限
        state_low = np.array([0] * 2 * (1 + act_num))  # 状态空间下限，包括密度差和动作
        state_high = np.array([1] * 2 * (1 + act_num))  # 状态空间上限，包括密度差和动作
        self.observation_space = Box(low=state_low, high=state_high, dtype=np.float32)  # 状态空间：Box类型，值范围为[0, 1]

    def reset(self):
        """
        重置 Jet 环境

        返回：
        - self.state: 环境初始状态
        """

        # 将输出密度重置为全0
        self.output_density = np.array([[0] * 2], dtype='float32')  # 输出密度初始化为零
        self.distance = self.target_density - self.output_density  # 计算目标密度与输出密度的差异
        self.action = np.array([[0] * hist_num * act_num], dtype='float32')  # 重置动作为零
        self.state = np.concatenate((self.distance, self.action), axis=1)  # 重设状态为密度差和动作的拼接

        return self.state  # 返回环境的初始状态

    def step(self, action, Voltage=0.8, Flow=0.2, O2=0,  H2o=0.1):
        """
        环境执行一步操作

        参数：
        - action: 由代理生成的控制动作
        - Voltage: 电压
        - O2: 氧气量
        - Flow: 流量
        - H2o: 水分量

        返回：
        - new_state: 执行动作后的新状态
        - reward: 当前动作的奖励
        - done: 是否结束该回合（episode）
        - info: 额外的信息（空字典）
        """
        terminal = False  # 默认非终止状态

        # 创建动作数组，action 是由代理计算出来的控制动作
        act = np.array([Voltage, O2, Flow, H2o]).reshape(1, 4)  # 将输入参数放入动作数组
        act[0, 0:act_num] = action  # 用提供的动作替换电压、氧气和水的前几个值

        # 使用模型进行预测，计算不同种类的密度
        density_o = max(np.squeeze(self.SysModel_O.predict(act)), 0)  # 预测O密度，取最大值防止负值
        density_no = max(np.squeeze(self.SysModel_NO.predict(act)), 0)  # 预测NO密度，取最大值防止负值
        density_oh = max(np.squeeze(self.SysModel_OH.predict(act)), 0)  # 预测OH密度，取最大值防止负值

        # 反归一化处理，将预测的密度值恢复到原始范围
        Pred_O = de_norm(density_o, YMin_Arr[0], YMax_Arr[0])  # 反归一化O密度
        Pred_NO = de_norm(density_no, YMin_Arr[1], YMax_Arr[1])  # 反归一化NO密度
        Pred_OH = de_norm(density_oh, YMin_Arr[2], YMax_Arr[2])  # 反归一化OH密度
        flow_dn = Flow * (5-0.3) + 0.3 #对气体流量反归一化
        # 更新输出ETOP密度
        self.output_density = np.roll(self.output_density, 1)  # 滚动输出ETOP密度数组，模拟新的状态
        self.output_density[0][0] =(Pred_O * 1e21 * 2.42 + Pred_NO * 1e20 * 1.59+ Pred_OH * 1e19 * 2.8 ) * flow_dn / 1e3 / 60  / 1e16   #计算ETOP/1e18


        # 更新动作，保存历史动作数据
        for i in range(hist_num - 1):  # 历史动作数据向后移动
            self.action[0, hist_num - i - 1:hist_num * act_num:hist_num] = self.action[0, hist_num - i - 2:hist_num * act_num:hist_num]
        self.action[0, 0:hist_num * act_num:hist_num] = action  # 保存当前动作

        # 重新计算状态：差距 + 动作
        self.distance = self.target_density - self.output_density
        self.state = np.concatenate((self.distance, self.action), axis=1)
        # 计算奖励：
        dis = abs(self.output_density[0][0] - self.target_density).sum()


        # 修改奖励函数
        k = 100  # 惩罚系数，可以根据需要调整
        reward = -k * dis  # 连续的惩罚函数

        # 设置一个阈值，当距离非常小时给予额外奖励
        if dis < 0.01:
            reward += 10  # 给予额外奖励
            terminal = True  # 可以根据需要决定是否结束当前 episode



        # 返回新的状态，奖励，是否结束，以及额外信息
        info = action  # 额外信息设置为当前的控制动作
        return self.state, reward, terminal, info  # 返回状态、奖励、终止标志、额外信息