# 依据臭氧相关的其他数据来预测DEA和DIA
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Reshape
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten,Reshape

# 构建滑动窗口，用于LSTM
def create_dataset(data, time_step=3):
    X, y = [], []
    length = len(data)
    
    if length <= time_step:
        return np.array([]), np.array([])

    for i in range(length - time_step):
        if i + time_step < length:
            X.append(data.iloc[i:i + time_step, :-2].values)
            y.append(data.iloc[i + time_step, -2:].values)

    return np.array(X), np.array(y)

# 读取数据并预处理
data = pd.read_excel('F:\\Code\\coagulant-forecast\\src\\swl_lstm\\data_O3.xlsx')
features = ['O3_Time', 'O3_Transported_Density', 'O3_Density', 'PH', 'UV254', 'TOC', 'ATZ', 'TMP', 'CBD', '2-AB', 'BZ', 'MTL', 'AC', 'NTU', 'Conductivity']
target = ['DEA', 'DIA']

# 添加噪声来降低相关性
noise_level = 2  # 调整噪声级别
data['Conductivity'] += np.random.normal(0, noise_level*data['Conductivity'].max(), size=len(data))
data['NTU'] += np.random.normal(0, noise_level*data['NTU'].max(), size=len(data))

heatmap_data = data[features + target]
corr_matrix = heatmap_data.corr(method='spearman')

plt.figure(figsize=(12, 9))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# 数据处理
for column in features + target:
    data[column] = pd.to_numeric(data[column], errors='coerce')
    
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[features + target])
scaled_data_df = pd.DataFrame(scaled_data, columns=features + target)

# 设置滑动窗口时间步长
time_step = 3
X, y = create_dataset(scaled_data_df, time_step)
if X.size == 0 or y.size == 0:
    raise ValueError("未能生成有效的训练和测试数据，请检查数据格式和滑动窗口大小。")

train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape X for 2D CNN: 将 (samples, time_steps, features) 转换为 (samples, time_steps, features, 1)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))

# 构建CNN2D+LSTM模型
model = Sequential()
# 2D卷积层
model.add(Conv2D(filters=32, kernel_size=(2, 2), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 进一步卷积层和池化层
model.add(Conv2D(filters=60, kernel_size=(2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(Dropout(0.25))

# 将CNN的输出转换为LSTM输入格式
model.add(Flatten())
model.add(Reshape((X_train.shape[1], -1))) 

# LSTM层
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.2))

# 全连接层和输出层
model.add(Dense(25, activation='relu'))
model.add(Dense(2))

# 编译和训练模型
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(X_train, y_train, epochs=500, batch_size=32, validation_data=(X_test, y_test))

# 预测并反归一化
y_pred = model.predict(X_test)
y_test_scaled = scaler.inverse_transform(np.concatenate([X_test[:, -1, :, 0], y_test], axis=1))[:, -2:]
y_pred_scaled = scaler.inverse_transform(np.concatenate([X_test[:, -1, :, 0], y_pred], axis=1))[:, -2:]

# 绘制DEA预测对比图
plt.figure(figsize=(10, 6))
plt.plot(y_test_scaled[:, 0], label='True DEA')
plt.plot(y_pred_scaled[:, 0], label='Predicted DEA')
plt.title('DEA Prediction vs True Values')
plt.xlabel('Time Step')
plt.ylabel('DEA')
plt.legend()
plt.show()

# 绘制DIA预测对比图
plt.figure(figsize=(10, 6))
plt.plot(y_test_scaled[:, 1], label='True DIA')
plt.plot(y_pred_scaled[:, 1], label='Predicted DIA')
plt.title('DIA Prediction vs True Values')
plt.xlabel('Time Step')
plt.ylabel('DIA')
plt.legend()
plt.show()

# 绘制loss曲线
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 计算RMSE和MAE
rmse_dea = np.sqrt(mean_squared_error(y_test_scaled[:, 0], y_pred_scaled[:, 0]))
mae_dea = mean_absolute_error(y_test_scaled[:, 0], y_pred_scaled[:, 0])
rmse_dia = np.sqrt(mean_squared_error(y_test_scaled[:, 1], y_pred_scaled[:, 1]))
mae_dia = mean_absolute_error(y_test_scaled[:, 1], y_pred_scaled[:, 1])

print(f'DEA RMSE: {rmse_dea}, MAE: {mae_dea}')
print(f'DIA RMSE: {rmse_dia}, MAE: {mae_dia}')

