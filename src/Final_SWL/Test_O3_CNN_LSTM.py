import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten,Reshape
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns

# 获取当前时间
current_time = datetime.now().strftime("%y%m%d%H%M%S")
path=f'.\\outputs\\CNN_LSTM\\{current_time}\\'
os.makedirs(path,exist_ok=True)
res=""

# 读取并准备数据
data = pd.read_excel('F:\\Code\\coagulant-forecast\\src\\swl_lstm\\data_O3.xlsx')
origin_features = ['O3_Time', 'O3_Transported_Density', 'O3_Density', 'PH', 'UV254', 'TOC', 'ATZ', 'TMP', 'CBD', '2-AB', 'BZ', 'NTU', 'Conductivity','DEA', 'DIA']
target =['UV254', 'TOC', 'ATZ', 'TMP', 'CBD', '2-AB', 'BZ','DEA', 'DIA']

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
features=['O3_Time', 'O3_Transported_Density', 'O3_Density', 'PH', 'UV254', 'TOC', 'ATZ', 'TMP', 'CBD', '2-AB', 'BZ','DEA', 'DIA']
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[features])
scaled_data_df = pd.DataFrame(scaled_data, columns=features)#135,13

# 构建滑动窗口
def create_dataset(data, time_step=3):
    X, y = [], []
    length = len(data)
    if length <= time_step:
        return np.array([]), np.array([])

    for i in range(length - time_step+1):
        
        # X包含所有属性列
        X.append(data.iloc[i:i + time_step-1, :].values)  
        # y仅包含目标属性列
        y.append(data.iloc[i + time_step-1, -9:].values)  

    return np.array(X), np.array(y)


def calculate_mape(y_true, y_pred):
    mask = y_true != 0
    return mean_absolute_percentage_error(y_true[mask], y_pred[mask])

time_step = 3
print(scaled_data_df.shape)
X, y = create_dataset(scaled_data_df, time_step)
#print(X.shape, y.shape)#(132, 3, 13) (132, 9)

# 设置 K 折交叉验证
k = 5  # 设置折数
kf = KFold(n_splits=k, shuffle=True, random_state=1)
fold = 1

# 保存每一折的评估结果
rmse_scores, mae_scores, mape_scores, r2_scores = [], [], [], []

# 所有的预测和真实值
all_preds = []
all_true = []


for train_index, test_index in kf.split(X):
    # 划分训练集和测试集
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    #print(X_train.shape, X_test.shape)#(106, 2, 13) (27, 2, 13)
    # 获取输入形状 (time_step, 特征数)
    input_shape = (X_train.shape[1], X_train.shape[2])#2,13
    
    # 构建1DCNN+LSTM模型
    model = Sequential()
    # CNN
    model.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Conv1D(filters=30, kernel_size=2, activation='relu'))#这里cnn滤波器数量应为X_train.shape[1]的倍数，也就是3的倍数，可以是60
    # 将输出展开为一维向量，以便输入 LSTM
    model.add(Reshape((X_train.shape[1], -1)))  # 重塑为 (time_steps, features)
    model.add(LSTM(32, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))  # 添加Dropout防止过拟合
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))  # 添加Dropout防止过拟合
    model.add(Dense(25, activation='relu'))
    model.add(Dense(9))  # 输出11-2个值，对应预测的目标属性

    # 编译模型
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # 训练模型
    history = model.fit(X_train, y_train, epochs=500, batch_size=32, validation_data=(X_test, y_test), verbose=0)
    
    # 预测
    y_pred = model.predict(X_test)#27 9,后面是26, 9
    print(y_pred.shape)
    # 反标准化预测值和真实值
    # 在反归一化时，为了确保维度匹配，将非目标列用零填充
    y_test_scaled = scaler.inverse_transform(np.concatenate([np.zeros((y_test.shape[0], X_test.shape[2]-y_test.shape[1])), y_test], axis=1))[:, -9:]
    y_pred_scaled = scaler.inverse_transform(np.concatenate([np.zeros((y_pred.shape[0], X_test.shape[2]-y_test.shape[1])), y_pred], axis=1))[:, -9:]
    
    all_preds.extend(y_pred_scaled)
    all_true.extend(y_test_scaled)

    # 计算每个指标
    for i, target_name in enumerate(target):
        rmse = np.sqrt(mean_squared_error(y_test_scaled[:, i], y_pred_scaled[:, i]))
        mae = mean_absolute_error(y_test_scaled[:, i], y_pred_scaled[:, i])
        mape = calculate_mape(y_test_scaled[:, i], y_pred_scaled[:, i])
        r2 = r2_score(y_test_scaled[:, i], y_pred_scaled[:, i])
        res += f"Fold {fold} - {target_name}: RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.4f}, R2={r2:.4f}\n"

        # 保存每一折的指标
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        mape_scores.append(mape)
        r2_scores.append(r2)
    
    fold += 1


# 转换为数组
all_preds = np.array(all_preds)
all_true = np.array(all_true)

# 绘制所有目标变量的预测对比图
for i, target_name in enumerate(target):
    plt.figure(figsize=(10, 6))
    plt.plot(all_true[:, i], label=f'True {target_name}')
    plt.plot(all_preds[:, i], label=f'Predicted {target_name}')
    plt.title(f'{target_name} Prediction vs True Values across 5-folds')
    plt.xlabel('Time Step')
    plt.ylabel(target_name)
    plt.legend()
    plt.savefig(path + f'\\Total_{target_name}_prediction_vs_true_values.png')

# 计算每个目标变量的平均指标
for i, target_name in enumerate(target):
    avg_rmse = np.mean([rmse_scores[i]])
    avg_mae = np.mean([mae_scores[i]])
    avg_mape = np.mean([mape_scores[i]])
    avg_r2 = np.mean([r2_scores[i]])
    res += f"\nAverage {target_name} RMSE: {avg_rmse:.4f}, MAE: {avg_mae:.4f}, MAPE: {avg_mape:.4f}, R2: {avg_r2:.4f}\n"


with open(path+"\\output.txt", "a", encoding="utf-8") as file:
    file.write(res + "\n")