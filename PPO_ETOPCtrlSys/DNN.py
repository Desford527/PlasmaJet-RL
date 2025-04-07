"""
该代码实现了基于神经网络的等离子体剂量ETOP（等效总氧化电位）预测模型。通过机器学习技术，模型使用输入的电气参数（如电压、流速、氧气含量等）来预测三种主要反应氧氮物种（RONS）：O、NO、OH的浓度。随后，这些浓度值被用于计算ETOP值。

代码流程包括：
1. 数据读取和归一化：从CSV文件中读取实验数据，并进行归一化处理，以便神经网络处理大尺度不同的输入数据。
2. 模型构建：通过构建多层全连接神经网络（使用ReLU激活函数），模型预测不同电气参数下的RONS浓度。
3. 模型训练：使用归一化后的数据训练模型，并评估模型在测试集上的性能。
4. ETOP计算：根据预测的RONS浓度计算ETOP值，评估等离子体剂量。
5. 结果可视化：绘制训练过程中的误差变化、预测值与真实值的比较，以及ETOP的计算结果。
"""

# 导入必要的库
import os  # Add this import
import matplotlib.pyplot as plt  # 用于绘制图形
import numpy as np  # 用于数值计算和数组操作
import pandas as pd  # 用于数据读取和处理
import csv
import scipy.stats as stats

from tensorflow import keras  # Keras高阶API，用于快速构建模型
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # 用于模型评估的指标
from DNN_Def import norm
from DNN_Def import de_norm
from DNN_Def import build_model
from DNN_Def import PrintDot
from DNN_Def import plot_history

from DNN_Constant import para
from DNN_Constant import Orig_data_path
from DNN_Constant import Train_file_path
# from DNN_Constant import Result_file_path
from DNN_Constant import Species_name
from DNN_Constant import Model_file_path

ROS_Num=3
YMax_Arr = []
YMin_Arr = []
XMax_Arr = []
XMin_Arr = []

Model = []


i = 0
#寻找数据集的统一最大与最小值
for i in range(ROS_Num):
    ROSIndex = i
    raw_dataset = pd.read_csv(Orig_data_path[ROSIndex], na_values="?", comment='\t', sep=",", skipinitialspace=True)
    dataset = raw_dataset.copy()  # 复制数据集
    dataset.pop(Species_name[ROSIndex])
    if i == 0:
        x_max = np.max(dataset, axis=0)
        x_min = np.min(dataset, axis=0)
    else:
        x_max = np.maximum(np.max(dataset, axis=0), x_max)
        x_min = np.minimum(np.min(dataset, axis=0), x_min)

### =================== Beginning of For Loop ===================
i = 0
for i in range(ROS_Num):
    ROSIndex = i

    # 读取CSV数据文件，处理缺失值和注释行
    raw_dataset = pd.read_csv(Orig_data_path[ROSIndex], na_values="?", comment='\t', sep=",", skipinitialspace=True)
    dataset = raw_dataset.copy()  # 复制数据集
    # 将数据集随机划分为训练集和测试集，80%用于训练，20%用于测试
    train_dataset = dataset.sample(frac=0.8, random_state=2)
    test_dataset = dataset.drop(train_dataset.index)


    # 分离标签（预测目标）和特征（输入数据）
    train_label = train_dataset.pop(Species_name[ROSIndex])  # 从训练集中移除标签列，并保存到train_labels
    test_label = test_dataset.pop(Species_name[ROSIndex])    # 从测试集中移除标签列，并保存到test_labels
    dataset.pop(Species_name[ROSIndex])


    # 获取标签的最大值和最小值，用于数据反归一化
    Spec_max= max(np.max(train_label), np.max(test_label))
    Spec_min = min(np.min(train_label), np.min(test_label))
    YMax_Arr.append(Spec_max)
    YMin_Arr.append(Spec_min)
    Subtr_max_min=Spec_max - Spec_min

    # Before saving files, ensure the directory exists
    os.makedirs(Train_file_path[ROSIndex], exist_ok=True)
    # 将训练集和测试集的数据和标签保存为CSV文件
    train_dataset.to_csv(Train_file_path[ROSIndex] + "train_data.csv", sep=',', header=False, index=False)
    test_dataset.to_csv(Train_file_path[ROSIndex] + "test_data.csv", sep=',', header=False, index=False)
    train_label.to_csv(Train_file_path[ROSIndex] + "train_label.csv", sep=',', header=False, index=False)
    test_label.to_csv(Train_file_path[ROSIndex] + "test_label.csv", sep=',', header=False, index=False)



    XMax_Arr.append(x_max)
    XMin_Arr.append(x_min)
    # 对训练数据和测试数据进行归一化处理
    normed_train_data = norm(train_dataset, x_min, x_max)
    normed_train_label = norm(train_label, YMin_Arr[ROSIndex], YMax_Arr[ROSIndex])
    normed_test_data = norm(test_dataset, x_min, x_max)
    normed_test_label = norm(test_label, YMin_Arr[ROSIndex], YMax_Arr[ROSIndex])

    # 构建和编译模型
    TempModel = build_model(para)
    Model.append(TempModel)

    EPOCHS = 5000  # 设置训练的总周期数
    # 设置早停回调函数，当验证集损失在50个周期内不再降低时停止训练
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=200)
    # 训练模型，使用归一化后的训练数据和标签，划分10%的数据作为验证集
    history = Model[ROSIndex].fit(normed_train_data, normed_train_label, epochs=EPOCHS,
                          validation_split=0.1, verbose=0, callbacks=[early_stop, PrintDot()])
    # 绘制训练过程中的损失和误差曲线
    plot_history(history, YMin_Arr[ROSIndex], YMax_Arr[ROSIndex], ROSIndex, Train_file_path[ROSIndex])
    # 保存训练好的模型
    # Model[ROSIndex].save( Result_file_path[ROSIndex] + ".keras")
    # 问题发生在试图保存模型时，系统提示：模型必须是 Functional 或 Sequential 类型，而你当前的模型可能是 Subclassed Model，这类模型不能被安全地序列化为 HDF5 格式。
    # Model[ROSIndex].save_weights(Result_file_path[ROSIndex] + "_weights.h5")
    Model[ROSIndex].save(Model_file_path[ROSIndex])

    # 在测试集上评估模型的性能
    loss, mae, mse = Model[ROSIndex].evaluate(normed_test_data, normed_test_label, verbose=2)
    # 使用模型对测试数据进行预测，并反归一化预测结果
    test_prediction = de_norm(Model[ROSIndex].predict(normed_test_data).flatten(), YMin_Arr[ROSIndex], YMax_Arr[ROSIndex])
    # 输出模型在测试集上的均方误差、平均绝对误差和R2得分
    print("Train Results for Reaction Species" + str(Species_name[ROSIndex]),"\n")
    print("Average squared difference between true and predicted values :", mean_squared_error(test_label, test_prediction),"\n"
          "Average absolute difference between true and predicted values:", mean_absolute_error(test_label, test_prediction),"\n"
          "R² score (Coefficient of Determination) : ", r2_score(test_label, test_prediction))
    # 将预测结果和真实标签保存为CSV文件
    predict_label = pd.DataFrame(np.array([test_prediction, test_label]).transpose())
    predict_label.to_csv(Train_file_path[ROSIndex] + "predict_label.csv", sep=',', header=False, index=False)
    # 绘制预测值与真实值的散点图

    plt.figure()
    plt.scatter(test_label, test_prediction)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('True Value and Predicted Value of ' + str(Species_name[ROSIndex]))
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    plt.plot([-800, 800], [-800, 800])  # 绘制参考线

### =================== End of For Loop ===================



# ==================保存 YMax_Arr, YMin_Arr, XMax_Arr, XMin_Arr 到 CSV 文件 ===================

max_min_file_path = "max_min_values.csv"
with open(max_min_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['YMax_Arr', 'YMin_Arr', 'XMax_Arr', 'XMin_Arr'])  # 写入标题行
    # 假设这四个数组的长度相等，逐行写入
    for i in range(len(YMax_Arr)):
        writer.writerow([YMax_Arr[i], YMin_Arr[i], XMax_Arr[i].tolist(), XMin_Arr[i].tolist()])


# ==================读取一组自定义输入参数集， 用于后面预测效果展示 ===================
Give_data_path = 'test_data2.csv'  # 测试数据文件路径
Give_data = pd.read_csv(Give_data_path, na_values="?", comment='\t', sep=",", skipinitialspace=True)

# 对测试数据进行归一化处理，并保存为CSV文件

ND_Give_data = norm(Give_data, x_min, x_max)
ND_Give_data.to_csv("ND_Give_data1.csv", sep=',', header=False, index=False)

# 绘制归一化后的测试数据
plt.figure()
# 遍历所有列并绘制对应的曲线
for column in ND_Give_data.columns:
    plt.plot(ND_Give_data[column], label=column)
# 添加图例、标题和标签
plt.legend()
plt.title('Normalized Data Plot')
plt.xlabel('Index')
plt.ylabel('Normalized Value')
#plt.show()

# =================== 使用模型进行预测，并反归一化预测结果 ===================
PredDen = []
i = 0
for i in range(ROS_Num):
    ROSIndex = i

    y_max = YMax_Arr[i]
    y_min = YMin_Arr[i]


    print(i, ', y_max=' ,y_max)
    print('y_min=', y_min)





    nPred_Data = Model[i].predict(ND_Give_data).flatten()
    filename = "NormPred_"
    ROS_Name = Species_name[ROSIndex]
    file_path = filename + str(ROS_Name) + ".csv"
    np.savetxt(file_path, nPred_Data, delimiter=",")  # 保存归一化的预测结果

    Pred_Data = de_norm(nPred_Data , y_min, y_max)  # 反归一化
    PredDen.append(Pred_Data)

    filename = "Pred_"
    ROS_Name = Species_name[ROSIndex]
    file_path = filename + str(ROS_Name) + ".csv"
    np.savetxt(file_path, Pred_Data, delimiter=",")  # 保存反归一化的预测结果

    if ROSIndex == 0:
        lineSy = 'o'
        CL = 'red'
    elif ROSIndex == 1:
        lineSy = 's'
        CL = 'blue'
    elif ROSIndex == 2:
        lineSy = '^'
        CL = 'green'

    plt.figure()
    plt.scatter(range(len(Pred_Data)), Pred_Data, label='Predict'+str(Species_name[ROSIndex]), color = CL, marker= lineSy)  # 绘制预测值
    plt.plot(range(len(Pred_Data)), Pred_Data, color = CL, marker= lineSy)  # 绘制预测值
    plt.legend()
    plt.title('Predicted RONS concentrations for the given parameters')
    plt.xlabel('Group number--'+str(Species_name[ROSIndex]))
    plt.ylabel('Value')
# 将预测结果绘制成散点图


# =================== 计算ETOP值  ETOP的计算公式，预测的浓度值乘以对应的系数和处理参数===================
ETOP = (PredDen[0] * 1e21 * 2.42 + PredDen[1] * 1e20 * 1.59 + PredDen[2] * 1e19 * 2.8) * Give_data.Flow / 1e3 / 60 * 120
# 其中，Flow单位为L/min，处理时间为120秒

# 绘制ETOP的散点图
plt.figure()
plt.scatter(range(len(ETOP)), ETOP, label='ETOP', marker='o')
plt.plot(range(len(ETOP)),  ETOP, linestyle = '--', marker='o')
plt.legend()
plt.title('Predicted ETOP for the given parameters')
plt.xlabel('Group number')
plt.ylabel('Value')

# 保存ETOP数据到CSV文件
np.savetxt("predict_ETOP.csv", ETOP, delimiter=",")



# =================== 直观相关性分析（归一化输入参数与原始ETOP） ===================

# 将原始 ETOP 添加到归一化后的输入数据 ND_Give_data
ND_Give_data['ETOP'] = ETOP  # 保留原始 ETOP 值

# 计算相关性矩阵并提取与 ETOP 的相关性列
correlation_with_etop = ND_Give_data.corr()['ETOP'].drop('ETOP')  # 排除 ETOP 与自身的相关性
correlation_with_etop_sorted = correlation_with_etop.sort_values(ascending=False)  # 按相关性大小排序

# 1. 绘制条形图，展示参数对 ETOP 的相关性大小
plt.figure(figsize=(10, 6))
correlation_with_etop_sorted.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Correlation Between Normalized Inputs and ETOP')
plt.ylabel('Correlation Coefficient')
plt.xlabel('Input Parameters')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("correlation_bar_chart.png")  # 保存为图片


# 2. 绘制特定输入参数与 ETOP 的散点图（选择几个最强和最弱相关的参数）
# 获取最强和最弱相关的参数
top_positive_param = correlation_with_etop_sorted.index[0]  # 最高正相关
top_negative_param = correlation_with_etop_sorted.index[-1]  # 最高负相关


# 保存详细的相关性结果到 CSV 文件
correlation_with_etop_sorted.to_csv("correlation_with_etop.csv", sep=',', header=["Correlation Coefficient"])





plt.show()