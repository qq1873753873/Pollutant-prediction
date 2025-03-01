# 依据臭氧相关的其他数据来预测DEA和DIA
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten,Reshape
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from datetime import datetime
# 获取当前时间
current_time = datetime.now().strftime("%y%m%d%H%M%S")

# 构建滑动窗口，用于lstm
def create_dataset(data, time_step=3):
    X, y = [], []
    length = len(data)
    
    # 如果数据长度小于time_step，跳过该序列
    if length <= time_step:
        return np.array([]), np.array([])

    # 使用滑动窗口生成样本
    for i in range(length - time_step):
        if i + time_step < length:
            X.append(data.iloc[i:i + time_step, :-2].values)  # 输入特征，去除目标列
            y.append(data.iloc[i + time_step, -2:].values)   # 输出目标 DEA 和 DIA

    return np.array(X), np.array(y)


# 读取Excel文件
data = pd.read_excel('F:\\Code\\coagulant-forecast\\src\\swl_lstm\\data_O3.xlsx') #135*15

features = ['O3_Time', 'O3_Transported_Density', 'O3_Density', 'PH', 'UV254', 'TOC', 'ATZ', 'TMP', 'CBD', '2-AB', 'BZ', 'MTL', 'AC', 'NTU', 'Conductivity']
target = ['DEA', 'DIA']

#绘制热力图，进行相关性分析
heatmap_data=data[features + target]
print(heatmap_data)
# 计算相关矩阵
corr_matrix = heatmap_data.corr()
# 绘制热力图
plt.figure(figsize=(12, 9))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.savefig(f'.\\outputs\\{current_time}热力图.png')

# TODO：更新列名，选择需要的特征和目标列
features=features
target=target

# 转换为float类型，处理NaN
for column in features + target:
    data[column] = pd.to_numeric(data[column], errors='coerce')  # 转换为 float，错误设置为 NaN

# 数据归一化
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[features + target])
scaled_data_df = pd.DataFrame(scaled_data, columns=features + target)


# 设置滑动窗口时间步长
time_step = 3
X, y = create_dataset(scaled_data_df, time_step)
# 检查是否生成了有效的数据
if X.size == 0 or y.size == 0:
    raise ValueError("未能生成有效的训练和测试数据，请检查数据格式和滑动窗口大小。")

# 划分训练集和测试集
train_size = int(0.5 * len(X))
train_size_virtual = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
X_train_virtual, X_test_virtual = X[:train_size_virtual], X[train_size_virtual:]
y_train_virtual, y_test_virtual = y[:train_size_virtual], y[train_size_virtual:]

print(X_train.shape[1])
print(X_train.shape[2])
# 构建1DCNN+LSTM模型
model = Sequential()
# CNN
model.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))#3*15
model.add(Conv1D(filters=30, kernel_size=2, activation='relu'))#这里cnn滤波器数量应为X_train.shape[1]的倍数，也就是3的倍数，可以是60
# 将输出展开为一维向量，以便输入 LSTM
model.add(Reshape((X_train.shape[1], -1)))  # 重塑为 (time_steps, features)

model.add(LSTM(32, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))  # 添加Dropout防止过拟合
model.add(LSTM(32, return_sequences=False))
model.add(Dropout(0.2))  # 添加Dropout防止过拟合
model.add(Dense(25, activation='relu'))
model.add(Dense(2))  # 输出2个值，分别对应DEA和DIA

# 编译模型
model.compile(optimizer='adam', loss='mean_squared_error')
#earlystop_callback = EarlyStopping(monitor='val_loss', min_delta=0.0001,patience=100)
# 训练模型
#history = model.fit(X_train, y_train, epochs=2000, batch_size=32, callbacks=earlystop_callback, validation_data=(X_test, y_test))
history = model.fit(X_train_virtual, y_train_virtual, epochs=1000, batch_size=32,  validation_data=(X_test, y_test))
# 预测
y_pred = model.predict(X_test)

# 反标准化预测值和真实值
# 使用 O3时间 列来辅助反标准化
y_test_scaled = scaler.inverse_transform(np.concatenate([X_test[:, -1, :], y_test], axis=1))[:, -2:]
y_pred_scaled = scaler.inverse_transform(np.concatenate([X_test[:, -1, :], y_pred], axis=1))[:, -2:]


# 绘制DEA预测对比图
plt.figure(figsize=(10, 6))
plt.plot(y_test_scaled[:, 0], label='True DEA')
plt.plot(y_pred_scaled[:, 0], label='Predicted DEA')
plt.title('DEA Prediction vs True Values')
plt.xlabel('Time Step')
plt.ylabel('DEA')
plt.legend()

plt.savefig(f'.\\outputs\\{current_time}DEA预测对比图.png')

# 绘制DIA预测对比图
plt.figure(figsize=(10, 6))
plt.plot(y_test_scaled[:, 1], label='True DIA')
plt.plot(y_pred_scaled[:, 1], label='Predicted DIA')
plt.title('DIA Prediction vs True Values')
plt.xlabel('Time Step')
plt.ylabel('DIA')
plt.legend()
plt.savefig(f'.\\outputs\\{current_time}DIA预测对比图.png')

# 绘制loss曲线
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'.\\outputs\\{current_time}Loss图.png')

# 计算RMSE和MAE
rmse_dea = np.sqrt(mean_squared_error(y_test_scaled[:, 0], y_pred_scaled[:, 0]))
mae_dea = mean_absolute_error(y_test_scaled[:, 0], y_pred_scaled[:, 0])
rmse_dia = np.sqrt(mean_squared_error(y_test_scaled[:, 1], y_pred_scaled[:, 1]))
mae_dia = mean_absolute_error(y_test_scaled[:, 1], y_pred_scaled[:, 1])
# 输出并保存
res=f'{current_time}:\n'
print(f'DEA RMSE: {rmse_dea}, MAE: {mae_dea}')
res+=f'DEA RMSE: {rmse_dea}, MAE: {mae_dea}\n'
print(f'DIA RMSE: {rmse_dia}, MAE: {mae_dia}')
res+=f'DIA RMSE: {rmse_dia}, MAE: {mae_dia}\n'
with open(".\\outputs\\output.txt", "a", encoding="utf-8") as file:
    file.write(res + "\n")

