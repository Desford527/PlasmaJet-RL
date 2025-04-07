# 定义一个自定义的神经网络模型类，继承自tf.keras.Model
# 导入必要的库
import matplotlib.pyplot as plt  # 用于绘制图形
import pandas as pd  # 用于数据读取和处理
import tensorflow as tf  # 用于构建和训练深度学习模型

from tensorflow import keras  # Keras高阶API，用于快速构建模型
from keras.layers import Dense, Flatten  # 导入全连接层和展平层
from DNN_Constant import lr_rate

class SyslModel(tf.keras.Model):
    """
    基于神经网络的ETOP预测模型，依赖电气输入参数来预测反应物物种浓度 (O, NO, OH)。
    这三种RONS浓度用于计算ETOP值.
    """
    def __init__(self, para):
        super(SyslModel, self).__init__()  # 初始化父类
        # para 是包含每层神经元数量的全局参数列表
        self.d1 = Dense(para[0], activation='relu')  # 第一层，全连接层，ReLU激活函数
        self.d2 = Dense(para[1], activation='relu')  # 第二层，全连接层，ReLU激活函数
        self.d3 = Dense(para[2], activation='relu')  # 第三层，全连接层，ReLU激活函数
        self.d4 = Dense(1)  # 输出层，全连接层，线性激活函数（默认）

    # 定义前向传播过程
    def call(self, inputs):
        """
        输入数据经过三层隐藏层，最后输出预测的RONS浓度值。
        这些浓度值随后将用于计算等效总氧化电位 (ETOP)。
        """
        x = self.d1(inputs)  # 输入经过第一层
        x = self.d2(x)       # 然后经过第二层
        x = self.d3(x)       # 然后经过第三层
        return self.d4(x)    # 最后经过输出层，得到预测结果

# 定义模型构建函数，返回一个编译后的模型实例
def build_model(para):
    """
    构建并编译模型。使用Adam优化器和均方误差作为损失函数。
    模型的目标是最小化均方误差（MSE），从而提高RONS浓度的预测精度。
    """
    model_stance = SyslModel(para)  # 实例化自定义模型
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_rate)  # 定义优化器，使用Adam优化算法
    # 编译模型，指定损失函数为均方误差，评估指标为平均绝对误差和均方误差
    model_stance.compile(loss='mse',
                         optimizer=optimizer,
                         metrics=['mae', 'mse'])
    return model_stance  # 返回构建好的模型

# 定义数据归一化函数，将数据缩放到0到1之间
def norm(x, x_min, x_max):
    """
    归一化数据，将输入参数归一化到0到1之间，以便神经网络更容易处理大尺度不同的输入数据。
    """
    return (x - x_min) / (x_max - x_min)

# 定义数据反归一化函数，将归一化的数据还原到原始范围
def de_norm(x, y_min, y_max):

    """
    反归一化函数，用于将神经网络预测的归一化值还原回原始的RONS浓度范围。
    """
    return y_min + x * (y_max - y_min)
#######这里y_min and y_max没有给出啊？！

# 自定义回调函数，用于在每个训练周期结束时打印一个点，显示训练进度
class PrintDot(keras.callbacks.Callback):
    """
    自定义回调函数，用于在训练过程中打印进度。每100个周期换行，其他周期输出一个点。
    有助于监控长时间的模型训练过程。
    """
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:  # 每经过100个周期，打印一个换行符
            print('')
        print('.', end='')  # 打印一个点，不换行

# 定义函数，用于绘制训练过程中的损失和误差变化曲线
def plot_history(history, y_min, y_max, ROSIndex, Train_file_path):
    """
    绘制模型训练过程中的损失和误差变化曲线，以便评估模型的训练效果。
    此过程有助于判断模型是否存在过拟合现象，以及训练是否足够。
    """
    hist = pd.DataFrame(history.history)  # 将训练历史记录转换为DataFrame
    hist['epoch'] = history.epoch  # 添加epoch列
    # 将mae和val_mae（验证集的mae）反归一化到原始数据范围

    """ ###这一部分画： mae和分别val_mae表示训练和验证数据的平均绝对误差（MAE）
    hist['mae'] = hist['mae'] * (y_max - y_min)            # Mean Absolute Error on the training data.
    hist['val_mae'] = hist['val_mae'] * (y_max - y_min)    # Mean Absolute Error on the validation data.
    plt.figure()
    plt.xlabel('Epoch')  # 设置x轴标签为Epoch
    if ROSIndex == 0:
        plt.ylabel('Mean Abs Error [O]')
    elif ROSIndex == 1:
        plt.ylabel('Mean Abs Error [NO]')
    elif ROSIndex == 2:
        plt.ylabel('Mean Abs Error [OH]')
    # 根据sel的值设置y轴标签，表示不同的预测目标

    # 绘制训练集和验证集的平均绝对误差曲线
    plt.plot(hist['epoch'], hist['mae'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],
             label='Val Error')
    plt.legend()  # 用于为图添加图例。图例是一个小框，用于解释图中使用的不同线条、标记或颜色的含义
    """
    ### 将mse和val_mse反归一化到原始数据范围. mse`（训练数据的均方误差); `val_mse`（验证数据的均方误差）：
    hist['mse'] = hist['mse'] * (y_max - y_min) ** 2          # Mean Squared Error on the training data
    hist['val_mse'] = hist['val_mse'] * (y_max - y_min) ** 2  # Mean Squared Error on the validation data.

    plt.figure()
    plt.xlabel('Epoch')
    # 根据sel的值设置y轴标签，表示不同的预测目标
    if ROSIndex == 0:  # 0表示O，1表示NO，2表示OH
        plt.ylabel('Mean Square Error [O]')
    elif ROSIndex == 1:
        plt.ylabel('Mean Square Error [NO]')
    elif ROSIndex == 2:
        plt.ylabel('Mean Square Error [OH]')
    # 绘制训练集和验证集的均方误差曲线
    plt.plot(hist['epoch'], hist['mse'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'], label='Val Error')
    plt.legend()
    # 将训练历史保存为CSV文件
    hist.to_csv(Train_file_path + "hist.csv", sep=',')