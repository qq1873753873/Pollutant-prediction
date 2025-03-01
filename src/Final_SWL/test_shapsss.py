import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

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

# 将 3D 数据转换为 2D
X_2d = X.reshape(X.shape[0], -1)  # 转换为 (n_samples, n_timesteps * n_features)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_2d, y, test_size=0.2, random_state=42)

# 训练随机森林模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 提取特征重要性
feature_importances = model.feature_importances_
print("Feature importances:", feature_importances)

# 将特征重要性和特征名称组合在一起
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
})

# 按重要性从高到低排序
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# 打印排序后的特征重要性
print("Sorted feature importances:")
print(feature_importance_df)

# 可视化特征重要性
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Random Forest Feature Importance (Sorted)")
plt.gca().invert_yaxis()  # 从高到低显示
plt.savefig("random_forest_feature_importance_sorted.png")
plt.show()