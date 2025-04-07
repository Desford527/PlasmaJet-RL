para = [512, 256, 128]  # 每一层的神经元数量

lr_rate = 0.001  # 学习率

column_names = ['Voltage', 'Flow', 'O2', 'H2o']  # 数据集的列名

Orig_data_path = ["Data_O.csv", "Data_NO.csv", "Data_OH.csv"]

Train_file_path = ["DNN_Data/Train_O/", "DNN_Data/Train_NO/", "DNN_Data/Train_OH/"]

# Result_file_path =["DNN_Data/DNN_Model_O", "DNN_Data/DNN_Model_NO", "DNN_Data/DNN_Model_OH"]

Model_file_path = ["DNN_Model/DNN_Model_O", "DNN_Model/DNN_Model_NO", "DNN_Model/DNN_Model_OH"]

Species_name = ["O", "NO", "OH"]  # 标签列名称