import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.layers import Dropout
# 读取Excel文件
data = pd.read_excel('F:\\Code\\coagulant-forecast\\src\\swl_lstm\\data.xls')

# 选择需要的特征和目标列
features = ['O3', 'GA', 'PH', 'NTU', 'UV254', 'OD680', '电导率', 'ATZ', '甲基硫菌灵', 
            '多菌灵', '2-氨基苯并咪唑', '苯并咪唑', '甲霜灵', '乙草胺', '臭氧浓度']
target = ['DEA', 'DIA']

for column in features + target:
    data[column] = pd.to_numeric(data[column], errors='coerce')  # 转换为 float，错误设置为 NaN
int_columns = data.select_dtypes(include=['int64']).columns
data[int_columns] = data[int_columns].astype(float)

# 根据特征和目标列划分数据
def detect_sequences(data, threshold=1e-3):
    sequences = []
    current_sequence = []
    start_idx = 0
    while start_idx < len(data):
        end_idx = start_idx
        current_sequence = [data.iloc[start_idx]]
        while end_idx + 1 < len(data) and (
            abs(data['O3'].iloc[end_idx + 1] - data['O3'].iloc[end_idx]) <= threshold or
            abs(data['GA'].iloc[end_idx + 1] - data['GA'].iloc[end_idx]) <= threshold):
            print(data['O3'].iloc[end_idx + 1])
            print(data['O3'].iloc[end_idx])
            end_idx += 1
            current_sequence.append(data.iloc[end_idx])
        
        sequences.append(pd.DataFrame(current_sequence))
        start_idx = end_idx + 1
    
    return sequences

# 划分数据
split_data = detect_sequences(data)

# 数据归一化
scaler = MinMaxScaler()
normalized_sequences = []

for seq in split_data:
    scaled_seq = scaler.fit_transform(seq[features + target])
    normalized_sequences.append(pd.DataFrame(scaled_seq, columns=features + target))

# 2. 构建滑动窗口
def create_dataset(data, time_step=3):
    X, y = [], []
    for series in data:
        length = len(series)
        if length <= time_step:
            continue
        
        # 使用滑动窗口生成样本
        for i in range(length - time_step):
            if i + time_step < length:
                X.append(series.iloc[i:i + time_step, :-2].values)  # 输入特征
                y.append(series.iloc[i + time_step, -2:].values)   # 输出目标

    return np.array(X), np.array(y)




# 设置滑动窗口时间步长
time_step = 1
X, y = create_dataset(normalized_sequences, time_step)

# 3. 划分训练集和测试集
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 4. 构建LSTM模型
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))  # 添加Dropout防止过拟合
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.2))  # 第二层Dropout
model.add(Dense(25, activation='relu'))
model.add(Dense(2))  # 输出2个值，分别对应DEA和DIA

# 编译模型
model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
history = model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_test, y_test))

# 5. 预测
y_pred = model.predict(X_test)

# 6. 反标准化预测值和真实值
y_test_scaled = scaler.inverse_transform(np.concatenate([X_test[:, -1, :], y_test], axis=1))[:, -2:]
y_pred_scaled = scaler.inverse_transform(np.concatenate([X_test[:, -1, :], y_pred], axis=1))[:, -2:]

# 7. 绘制DEA预测对比图
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

# 8. 绘制loss曲线
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 9. 计算RMSE和MAE
rmse_dea = np.sqrt(mean_squared_error(y_test_scaled[:, 0], y_pred_scaled[:, 0]))
mae_dea = mean_absolute_error(y_test_scaled[:, 0], y_pred_scaled[:, 0])
rmse_dia = np.sqrt(mean_squared_error(y_test_scaled[:, 1], y_pred_scaled[:, 1]))
mae_dia = mean_absolute_error(y_test_scaled[:, 1], y_pred_scaled[:, 1])

print(f'DEA RMSE: {rmse_dea}, MAE: {mae_dea}')
print(f'DIA RMSE: {rmse_dia}, MAE: {mae_dia}')
