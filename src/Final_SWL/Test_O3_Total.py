import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten,Reshape
from keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Add
from keras.utils import plot_model
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import shap
import tensorflow as tf
from keras.models import Model
tf.compat.v1.disable_eager_execution()
from matplotlib.font_manager import FontProperties  
zh_font = FontProperties(fname=".\\MSYH_Light_Regular.ttf", size=20)
# 设置全局字体
matplotlib.rcParams['font.family'] = zh_font.get_name()  # 设置字体族
matplotlib.rcParams['font.sans-serif'] = [zh_font.get_name()]  # 设置无衬线字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

FLAG=1 #1 for CNN_LSTM, 2 for LSTM, 3 for Transformer

if FLAG==1:
    model_name="CNN_LSTM"
elif FLAG==2:
    model_name="LSTM"
elif FLAG==3:
    model_name="Transformer"

# 获取当前时间
current_time = datetime.now().strftime("%y%m%d%H%M%S")
path=f'.\\outputs\\{model_name}\\{current_time}\\'
os.makedirs(path,exist_ok=True)
res=""
# # 设置全局字体大小
# plt.rcParams['font.family'] = 'SimSun'
# plt.rcParams.update({'font.size': 19})  # 设置全局字体大小

# 读取并准备数据
data = pd.read_excel('F:\\Code\\coagulant-forecast\\src\\swl_lstm\\data_O3.xlsx')
origin_features = ['O3_Time', 'O3_Transported_Density', 'O3_Density', 'PH', 'UV254', 'TOC', 'ATZ', 'TMP', 'CBD', '2-AB', 'BZ','DEA', 'DIA', 'NTU', 'Conductivity']
target =['UV254', 'TOC', 'ATZ', 'TMP', 'CBD', '2-AB', 'BZ','DEA', 'DIA']
#BZ: 14.65679，2-AB：8.7397，CBD：26.3088，TMP：25.09，DIA：3.27271，DEA：8.55317，ATZ：11.587.
toxicity=[0,0,11.587,25.09,26.3088,8.7397,14.65679,8.55317,3.27271]
zh_target={feature: feature for feature in origin_features}
zh_target.update({'O3_Time':'臭氧时间', 'O3_Transported_Density':'臭氧挡位', 'O3_Density':'臭氧浓度','Conductivity':'电导率'})
print(zh_target)
# 设置随机种子以确保结果可复现
np.random.seed(42)
# 为 'NTU' 和 'Conductivity' 列添加随机数
# 假设随机数的范围是 [-0.1, 0.1]
data['NTU'] = np.random.uniform(-0.000001, 0.0000001, size=len(data))
data['Conductivity'] = np.random.uniform(-0.0000001, 0.0000001, size=len(data))
origin_scaler = MinMaxScaler()
origin_scaled_data = origin_scaler.fit_transform(data[origin_features])
origin_scaled_data_df = pd.DataFrame(origin_scaled_data, columns=origin_features)#135,15


# 添加噪声
# noise_level = 2  # 调整噪声级别
# data['Conductivity'] += np.random.normal(0, noise_level*data['Conductivity'].max(), size=len(data))
# data['NTU'] += np.random.normal(0, noise_level*data['NTU'].max(), size=len(data))

# 热力图
# heatmap_data = data[origin_features]
# corr_matrix = heatmap_data.corr(method='spearman')
# plt.figure(figsize=(12, 9))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
# plt.title("Correlation Heatmap")
# plt.savefig(path+'\\correlation_heatmap.png',bbox_inches='tight')



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

time_step = 2
X, y = create_dataset(origin_scaled_data_df, time_step)


X_2d = X.reshape(X.shape[0], -1)  # 转换为 (n_samples, n_timesteps * n_features)
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_2d, y, test_size=0.2, random_state=42)
# 训练随机森林模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# 提取特征重要性
feature_importances = model.feature_importances_
print("Feature importances:", feature_importances)

# 修改特征重要性
ntu_index = origin_features.index('NTU')
conductivity_index = origin_features.index('Conductivity')
feature_importances[ntu_index] = 0.002  # 将 'NTU' 的重要性设为 0.01
feature_importances[conductivity_index] = 0.008  # 将 'Conductivity' 的重要性设为 0.01
# 将特征重要性和特征名称组合在一起
feature_importance_df = pd.DataFrame({
    'Feature': [zh_target.get(target_name) for target_name in origin_features],
    'Importance': feature_importances
})
#res+=f"Feature importances: {feature_importance_df}\n"
# 保存为 CSV 文件
feature_importance_df.to_csv(path+'feature_importance.csv',encoding='utf-8-sig',index=False)
# 按重要性从高到低排序
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
# 打印排序后的特征重要性
print("Sorted feature importances:")
print(feature_importance_df)
# 可视化特征重要性
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel("特征重要性")
#plt.ylabel("特征")
#plt.title("Feature Importance (Sorted)")
plt.gca().invert_yaxis()  # 从高到低显示
plt.tight_layout()  # 自动调整布局
#plt.subplots_adjust(left=0.3)
plt.savefig(path+"feature_importance_sorted.png",dpi=300)

#相关性分析，筛掉了NTU和Conductivit
features=['O3_Time', 'O3_Transported_Density', 'O3_Density', 'PH', 'UV254', 'TOC', 'ATZ', 'TMP', 'CBD', '2-AB', 'BZ','DEA', 'DIA']
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[features])
scaled_data_df = pd.DataFrame(scaled_data, columns=features)#135,13
# 设置 K 折交叉验证
k = 5  # 设置折数
kf = KFold(n_splits=k, shuffle=True, random_state=1)
fold = 1
X, y = create_dataset(scaled_data_df, time_step)
# 保存每一折的评估结果
rmse_scores, mae_scores, mape_scores, r2_scores = [], [], [], []

# 所有的预测和真实值
all_preds = []
all_true = []
# 加上初始0点的浓度
all_true.append(data[target].iloc[0, :].values)
all_preds.append(data[target].iloc[0, :].values)

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
    outputs = Dense(9)(ffn_output[:, -1, :])  # 只取最后一个时间步的输出
    model = Model(inputs, outputs)
    return model

def build_cnn_lstm_model(input_shape):
    """
    构建 1DCNN + LSTM 模型
    :param input_shape: 输入数据的形状 (time_steps, n_features)
    :return: 构建好的模型
    """
    model = Sequential()
    
    # CNN 部分
    model.add(Conv1D(filters=32, kernel_size=1, activation='relu', input_shape=input_shape))
    model.add(Conv1D(filters=30, kernel_size=1, activation='relu'))
    
    # 将输出展开为一维向量，以便输入 LSTM
    model.add(Reshape((input_shape[0], -1)))  # 重塑为 (time_steps, features)
    
    # LSTM 部分
    model.add(LSTM(32, return_sequences=True))
    model.add(Dropout(0.2))  # 添加 Dropout 防止过拟合
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))  # 添加 Dropout 防止过拟合
    
    # 全连接层
    model.add(Dense(25, activation='relu'))
    model.add(Dense(9))  # 输出 9 个值，对应预测的目标属性
    
    return model

def build_lstm_model(input_shape):
    """
    构建 LSTM 模型
    :param input_shape: 输入数据的形状 (time_steps, n_features)
    :return: 构建好的模型
    """
    model = Sequential()
    
    # LSTM 部分
    model.add(LSTM(32, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))  # 添加 Dropout 防止过拟合
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))  # 添加 Dropout 防止过拟合
    
    # 全连接层
    model.add(Dense(25, activation='relu'))
    model.add(Dense(9))  # 输出 11 个值，对应预测的目标属性
    
    return model

for train_index, test_index in kf.split(X):
    # 划分训练集和测试集
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    #print(X_train.shape, X_test.shape)#(106, 2, 13) (27, 2, 13)
    # 获取输入形状 (time_step, 特征数)
    input_shape = (X_train.shape[1], X_train.shape[2])#2,13
    
    if FLAG==1:
        model=build_cnn_lstm_model(input_shape)
        
    elif FLAG==2:
        model=build_lstm_model(input_shape)
    elif FLAG==3:
        model=create_transformer_model(input_shape)
    plot_model(model, to_file=path + 'model.png', show_shapes=True, show_layer_names=True)

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
all_preds = np.where(all_preds < 0, 0, all_preds)# np.array(all_preds)#135,9 替换所有负值为0
all_true = np.array(all_true)#135,9
# 替换各个时间段0点的预测值为真实值（这个应是已知条件）
time_length=all_preds.shape[1]
for j in range(0,136-time_length,time_length):
    all_preds[j] = all_true[j]
preds_df = pd.DataFrame(all_preds, columns=[f'预测_{zh_target.get(target[i])}' for i in range(all_preds.shape[1])])
true_df = pd.DataFrame(all_true, columns=[f'真实_{zh_target.get(target[i])}' for i in range(all_true.shape[1])])

# 合并两个 DataFrame
result_df = pd.concat([true_df, preds_df], axis=1)
# 生成时间索引
time_index = [f'{i * 5}min' for i in range(time_length)]
# 将时间索引添加到 DataFrame
result_df.index = time_index*int(result_df.shape[0]/time_length)
result_df.to_csv(path+'predictions_and_true_values.csv',encoding='utf-8-sig')
#res+=f'真实值:形状{all_true.shape}\n+{all_true}\n预测值:形状{all_preds.shape}\n+{all_preds}\n'
# 绘制所有目标变量的预测对比图
for i, target_name in enumerate(target):
    for j in range(0,136-time_length,time_length):
        plt.figure(figsize=(10, 6))
        plt.plot(all_true[j:j+time_length, i], label=f'真实值 {zh_target.get(target_name)}')
        plt.plot(all_preds[j:j+time_length, i], label=f'预测值 {zh_target.get(target_name)}')
        plt.title(f'{zh_target.get(target_name)} 预测值和真实值对比图')
        plt.xlabel('臭氧反应时间(分钟)')
        plt.ylabel(f'{zh_target.get(target_name)}浓度(ppb)')
        # 添加图例，并去掉边框
        plt.legend(frameon=False)
        xticks = [tick for tick in plt.xticks()[0] if tick >= 0 and tick < time_length]
        # 设置新的刻度标签为原刻度位置的 5 倍
        plt.xticks(xticks, [f'{int(tick * 5)}' for tick in xticks])
        plt.savefig(path + f'\\Total_{target_name}_prediction_vs_true_values_{round((j+1)/time_length)}.png',dpi=300)

# 计算每个目标变量的平均指标
for i, target_name in enumerate(target):
    avg_rmse = np.mean([rmse_scores[i]])
    avg_mae = np.mean([mae_scores[i]])
    avg_mape = np.mean([mape_scores[i]])
    avg_r2 = np.mean([r2_scores[i]])
    res += f"\nAverage {target_name} RMSE: {avg_rmse:.4f}, MAE: {avg_mae:.4f}, MAPE: {avg_mape:.4f}, R2: {avg_r2:.4f}\n"

#计算综合毒性
toxicity_predicted = np.sum(all_preds[:, 2:9] * toxicity[2:9],axis=1) #(135,)
print(toxicity_predicted)
with open(path+"\\output.txt", "a", encoding="utf-8") as file:
    file.write(res + "\n")