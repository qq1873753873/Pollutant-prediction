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
import seaborn as sns

# 获取当前时间
current_time = datetime.now().strftime("%y%m%d%H%M%S")
path=f'.\\outputs\\LSTM\\{current_time}\\'
os.makedirs(path,exist_ok=True)
res=""

# 读取并准备数据
data = pd.read_excel('F:\\Code\\coagulant-forecast\\src\\swl_lstm\\data_O3.xlsx')
origin_features = ['O3_Time', 'O3_Transported_Density', 'O3_Density', 'PH', 'UV254', 'TOC', 'ATZ', 'TMP', 'CBD', '2-AB', 'BZ', 'MTL', 'AC', 'NTU', 'Conductivity','DEA', 'DIA']

# 添加噪声
noise_level = 2  # 调整噪声级别
data['Conductivity'] += np.random.normal(0, noise_level*data['Conductivity'].max(), size=len(data))
data['NTU'] += np.random.normal(0, noise_level*data['NTU'].max(), size=len(data))

heatmap_data = data[origin_features]
corr_matrix = heatmap_data.corr(method='spearman')

plt.figure(figsize=(12, 9))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.savefig(path+'\\correlation_heatmap.png',bbox_inches='tight')

#相关性分析，筛掉了NTU和Conductivity
features=['O3_Time', 'O3_Transported_Density', 'O3_Density', 'PH', 'UV254', 'TOC', 'ATZ', 'TMP', 'CBD', '2-AB', 'BZ', 'MTL', 'AC','DEA', 'DIA']
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[features])
scaled_data_df = pd.DataFrame(scaled_data, columns=features)

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

time_step = 3
X, y = create_dataset(scaled_data_df, time_step)#132,因为时间步的问题

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
    
    #2 构建LSTM模型
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))#60改成32
    model.add(Dropout(0.2))  # 添加Dropout防止过拟合
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))  # 添加Dropout防止过拟合
    model.add(Dense(25, activation='relu'))
    model.add(Dense(2))  # 输出2个值，分别对应DEA和DIA

    # 编译模型
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # 训练模型
    history = model.fit(X_train, y_train, epochs=500, batch_size=32, validation_data=(X_test, y_test), verbose=0)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 反标准化预测值和真实值
    y_test_scaled = scaler.inverse_transform(np.concatenate([X_test[:, -1, :], y_test], axis=1))[:, -2:]
    y_pred_scaled = scaler.inverse_transform(np.concatenate([X_test[:, -1, :], y_pred], axis=1))[:, -2:]
    
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