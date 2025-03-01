import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Add
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 读取Excel文件
data = pd.read_excel('F:\\Code\\coagulant-forecast\\src\\swl_lstm\\data_O3.xlsx')
print(data.columns)

# 更新列名，选择需要的特征和目标列
features = ['O3_Time', 'O3_Transported_Density', 'O3_Density', 'PH', 'UV254', 'TOC', 'ATZ', 'TMP', 'CBD', '2-AB', 'BZ', 'MTL', 'AC', 'NTU', 'Conductivity']
target = ['DEA', 'DIA']

# 转换为float类型，处理NaN
for column in features + target:
    data[column] = pd.to_numeric(data[column], errors='coerce')  # 转换为 float，错误设置为 NaN

# 1. 数据归一化
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[features + target])
scaled_data_df = pd.DataFrame(scaled_data, columns=features + target)

# 2. 构建滑动窗口
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

# 设置滑动窗口时间步长
time_step = 3
X, y = create_dataset(scaled_data_df, time_step)

# 检查是否生成了有效的数据
if X.size == 0 or y.size == 0:
    raise ValueError("未能生成有效的训练和测试数据，请检查数据格式和滑动窗口大小。")

# 3. 划分训练集和测试集
train_size = int(0.5 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 4. 构建Transformer模型
def create_transformer_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # Multi-Head Attention
    attention_output = MultiHeadAttention(num_heads=4, key_dim=2)(inputs, inputs)
    attention_output = Add()([inputs, attention_output])  # 残差连接
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output)
    
    # Feed Forward
    ffn = Dense(64, activation='relu')(attention_output)
    ffn = Dense(input_shape[-1])(ffn)  # 输出维度与输入一致
    ffn_output = Add()([attention_output, ffn])  # 残差连接
    ffn_output = LayerNormalization(epsilon=1e-6)(ffn_output)

    # 输出层
    outputs = Dense(2)(ffn_output[:, -1, :])  # 只取最后一个时间步的输出
    model = Model(inputs, outputs)
    return model

# 创建模型
model = create_transformer_model((X_train.shape[1], X_train.shape[2]))

# 编译模型
model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
history = model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_test, y_test))

# 5. 预测
y_pred = model.predict(X_test)

# 6. 反标准化预测值和真实值
y_test_scaled = scaler.inverse_transform(np.concatenate([X_test[:, -1, :], y_test], axis=1))[:, -2:]
y_pred_scaled = scaler.inverse_transform(np.concatenate([X_test[:, -1, :], y_pred], axis=1))[:, -2:]
y_pred_scaled[y_pred_scaled < 0] = 0

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
