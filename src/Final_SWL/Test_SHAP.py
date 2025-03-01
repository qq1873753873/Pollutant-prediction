import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Reshape
import matplotlib.pyplot as plt
from datetime import datetime
import shap
import tensorflow as tf

# 禁用 TensorFlow 的 eager execution
tf.compat.v1.disable_eager_execution()

# 获取当前时间
current_time = datetime.now().strftime("%y%m%d%H%M%S")
path = f'.\\outputs\\CNN_LSTM\\{current_time}\\'
os.makedirs(path, exist_ok=True)

# 读取并准备数据
data = pd.read_excel('F:\\Code\\coagulant-forecast\\src\\swl_lstm\\data_O3.xlsx')
origin_features = ['O3_Time', 'O3_Transported_Density', 'O3_Density', 'PH', 'UV254', 'TOC', 'ATZ', 'TMP', 'CBD', '2-AB', 'BZ', 'NTU', 'Conductivity', 'DEA', 'DIA']
target = ['UV254', 'TOC', 'ATZ', 'TMP', 'CBD', '2-AB', 'BZ', 'DEA', 'DIA']

# 添加噪声
noise_level = 2  # 调整噪声级别
data['Conductivity'] += np.random.normal(0, noise_level * data['Conductivity'].max(), size=len(data))
data['NTU'] += np.random.normal(0, noise_level * data['NTU'].max(), size=len(data))

# 相关性分析，筛掉了 NTU 和 Conductivity
features = ['O3_Time', 'O3_Transported_Density', 'O3_Density', 'PH', 'UV254', 'TOC', 'ATZ', 'TMP', 'CBD', '2-AB', 'BZ', 'DEA', 'DIA']
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[features])
scaled_data_df = pd.DataFrame(scaled_data, columns=features)  # 135, 13

# 构建滑动窗口
def create_dataset(data, time_step=3):
    X, y = [], []
    length = len(data)
    if length <= time_step:
        return np.array([]), np.array([])

    for i in range(length - time_step + 1):
        # X 包含所有属性列
        X.append(data.iloc[i:i + time_step - 1, :].values)
        # y 仅包含目标属性列
        y.append(data.iloc[i + time_step - 1, -9:].values)

    return np.array(X), np.array(y)

time_step = 2
X, y = create_dataset(scaled_data_df, time_step)
print("X shape:", X.shape)  # (134, 1, 13)
print("y shape:", y.shape)  # (134, 9)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建初始模型用于 SHAP 分析
input_shape = (X_train.shape[1], X_train.shape[2])  # (1, 13)
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=1, activation='relu', input_shape=input_shape))
model.add(Conv1D(filters=30, kernel_size=1, activation='relu'))
model.add(Reshape((X_train.shape[1], -1)))  # 重塑为 (time_steps, features)
model.add(LSTM(32, return_sequences=True, input_shape=input_shape))
model.add(Dropout(0.2))
model.add(LSTM(32, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(25, activation='relu'))
model.add(Dense(9))  # 输出 9 个结果
model.compile(optimizer='adam', loss='mean_squared_error')

# 训练初始模型
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()
# SHAP 分析
explainer = shap.DeepExplainer(model, X_train[:100])  # 使用前 100 个样本作为背景数据
shap_values = explainer.shap_values(X_test, check_additivity=False)

# 检查 shap_values 的形状
print("shap_values shape:", np.array(shap_values).shape)  # 应该是 (n_outputs, n_samples, n_timesteps, n_features)

# 选择单个输出的 SHAP 值（例如第一个输出）
shap_values_single_output = shap_values[0]  # 选择第一个输出的 SHAP 值
print("shap_values_single_output shape:", shap_values_single_output.shape)  # 应该是 (n_samples, n_timesteps, n_features)

# 将 X_test 转换为 2D
X_test_2d = X_test.reshape(-1, X_test.shape[2])  # 转换为 (n_samples, n_features)
print("X_test_2d shape:", X_test_2d.shape)  # 应该是 (n_samples, n_features)

# 绘制单个输出的 SHAP 图
shap.summary_plot(shap_values_single_output.reshape(-1, X_test.shape[2]), X_test_2d, feature_names=features, plot_type="bar", show=False)
plt.title("SHAP Summary Plot (Single Output)")
plt.savefig(path + "SHAP_summary_plot_single_output.png")
plt.close()

# 对所有输出的 SHAP 值进行平均
shap_values_avg = np.mean(shap_values, axis=0)  # 对所有输出的 SHAP 值进行平均
print("shap_values_avg shape:", shap_values_avg.shape)  # 应该是 (n_samples, n_timesteps, n_features)

# 绘制平均 SHAP 图
shap.summary_plot(shap_values_avg.reshape(-1, X_test.shape[2]), X_test_2d, feature_names=features, plot_type="bar", show=False)
plt.title("SHAP Summary Plot (Averaged Across Outputs)")
plt.savefig(path + "SHAP_summary_plot_avg.png")
plt.close()
y_pred = model.predict(X_test)
print("Model predictions:", y_pred)