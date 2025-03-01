#只用来测DEA、DIA
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from datetime import datetime
from keras.models import Model
from keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Add

# 获取当前时间
current_time = datetime.now().strftime("%y%m%d%H%M%S")
path=f'.\\outputs\\Transformer\\{current_time}\\'
os.makedirs(path,exist_ok=True)
res=""

# 读取并准备数据
data = pd.read_excel('F:\\Code\\coagulant-forecast\\src\\swl_lstm\\data_O3.xlsx')
features = ['O3_Time', 'O3_Transported_Density', 'O3_Density', 'PH', 'UV254', 'TOC', 'ATZ', 'TMP', 'CBD', '2-AB', 'BZ', 'MTL', 'AC', 'NTU', 'Conductivity']
target = ['DEA', 'DIA']
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[features + target])
scaled_data_df = pd.DataFrame(scaled_data, columns=features + target)

# 构建滑动窗口
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

def calculate_mape(y_true, y_pred):
    mask = y_true != 0
    return mean_absolute_percentage_error(y_true[mask], y_pred[mask])

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


time_step = 3
X, y = create_dataset(scaled_data_df, time_step)

# 设置 K 折交叉验证
k = 5  # 设置折数
kf = KFold(n_splits=k, shuffle=True, random_state=1)
fold = 1

# 保存每一折的评估结果
rmse_scores, mae_scores, mape_scores, r2_scores = [], [], [], []

# 所有的预测和真实值
all_preds = []
all_true = []


# K 折交叉验证
for train_index, test_index in kf.split(X):
    # 划分训练集和测试集
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # 创建模型
    model = create_transformer_model((X_train.shape[1], X_train.shape[2]))

    # 编译模型
    model.compile(optimizer='adam', loss='mean_squared_error')

    # 训练模型
    history = model.fit(X_train, y_train, epochs=500, batch_size=32, validation_data=(X_test, y_test))

    # 预测
    y_pred = model.predict(X_test)
    
    # 反标准化预测值和真实值
    y_test_scaled = scaler.inverse_transform(np.concatenate([X_test[:, -1, :], y_test], axis=1))[:, -2:]
    y_pred_scaled = scaler.inverse_transform(np.concatenate([X_test[:, -1, :], y_pred], axis=1))[:, -2:]
    y_pred_scaled[y_pred_scaled < 0] = 0

    all_preds.extend(y_pred_scaled)
    all_true.extend(y_test_scaled)


    # 计算每个指标
    rmse_dea = np.sqrt(mean_squared_error(y_test_scaled[:, 0], y_pred_scaled[:, 0]))
    mae_dea = mean_absolute_error(y_test_scaled[:, 0], y_pred_scaled[:, 0])
    mape_dea = calculate_mape(y_test_scaled[:, 0], y_pred_scaled[:, 0])
    r2_dea = r2_score(y_test_scaled[:, 0], y_pred_scaled[:, 0])

    rmse_dia = np.sqrt(mean_squared_error(y_test_scaled[:, 1], y_pred_scaled[:, 1]))
    mae_dia = mean_absolute_error(y_test_scaled[:, 1], y_pred_scaled[:, 1])
    mape_dia = calculate_mape(y_test_scaled[:, 1], y_pred_scaled[:, 1])
    r2_dia = r2_score(y_test_scaled[:, 1], y_pred_scaled[:, 1])

     # 打印每一折的结果
    print(f"Fold {fold}: DEA RMSE: {rmse_dea}, DEA MAE: {mae_dea}, DEA MAPE: {mape_dea}, DEA R2: {r2_dea}")
    res+=f"Fold {fold}: DEA RMSE: {rmse_dea}, DEA MAE: {mae_dea}, DEA MAPE: {mape_dea}, DEA R2: {r2_dea}\n"
    print(f"Fold {fold}: DIA RMSE: {rmse_dia}, DIA MAE: {mae_dia}, DIA MAPE: {mape_dia}, DIA R2: {r2_dia}")
    res+=f"Fold {fold}: DIA RMSE: {rmse_dia}, DIA MAE: {mae_dia}, DIA MAPE: {mape_dia}, DIA R2: {r2_dia}\n"
    
    
    # 保存每一折的指标
    rmse_scores.append((rmse_dea, rmse_dia))
    mae_scores.append((mae_dea, mae_dia))
    mape_scores.append((mape_dea, mape_dia))
    r2_scores.append((r2_dea, r2_dia))
    fold += 1

# 转换为数组
all_preds = np.array(all_preds)
all_true = np.array(all_true)

# 绘制DEA预测对比图
plt.figure(figsize=(10, 6))
plt.plot(all_true[:, 0], label='True DEA')
plt.plot(all_preds[:, 0], label='Predicted DEA')
plt.title('DEA Prediction vs True Values across 5-folds')
plt.xlabel('Time Step')
plt.ylabel('DEA')
plt.legend()
plt.savefig(path+'\\Total_DEA_prediction_vs_true_values.png')

# 绘制DIA预测对比图
plt.figure(figsize=(10, 6))
plt.plot(all_true[:, 1], label='True DIA')
plt.plot(all_preds[:, 1], label='Predicted DIA')
plt.title('DIA Prediction vs True Values across 5-folds')
plt.xlabel('Time Step')
plt.ylabel('DIA')
plt.legend()
plt.savefig(path+'\\Total_DIA_prediction_vs_true_values.png')

# 打印每个指标的平均结果
avg_rmse_dea = np.mean([score[0] for score in rmse_scores])
avg_rmse_dia = np.mean([score[1] for score in rmse_scores])
avg_mae_dea = np.mean([score[0] for score in mae_scores])
avg_mae_dia = np.mean([score[1] for score in mae_scores])
avg_mape_dea = np.mean([score[0] for score in mape_scores])
avg_mape_dia = np.mean([score[1] for score in mape_scores])
avg_r2_dea = np.mean([score[0] for score in r2_scores])
avg_r2_dia = np.mean([score[1] for score in r2_scores])

print(f"\nAverage DEA RMSE: {avg_rmse_dea}, DEA MAE: {avg_mae_dea}, DEA MAPE: {avg_mape_dea}, DEA R2: {avg_r2_dea}")
res+=f"Average DEA RMSE: {avg_rmse_dea}, DEA MAE: {avg_mae_dea}, DEA MAPE: {avg_mape_dea}, DEA R2: {avg_r2_dea}\n"
print(f"Average DIA RMSE: {avg_rmse_dia}, DIA MAE: {avg_mae_dia}, DIA MAPE: {avg_mape_dia}, DIA R2: {avg_r2_dia}")
res+=f"Average DIA RMSE: {avg_rmse_dia}, DIA MAE: {avg_mae_dia}, DIA MAPE: {avg_mape_dia}, DIA R2: {avg_r2_dia}\n"

with open(path+"\\output.txt", "a", encoding="utf-8") as file:
    file.write(res + "\n")