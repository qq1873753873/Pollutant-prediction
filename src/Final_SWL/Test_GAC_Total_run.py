import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten,Reshape
from keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Add
from keras.models import load_model
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
# matplotlib.rcParams['font.family'] = zh_font.get_name()  # 设置字体族
# matplotlib.rcParams['font.sans-serif'] = [zh_font.get_name()]  # 设置无衬线字体
# matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 设置中文字体为宋体
plt.rcParams['font.sans-serif'] = ['SimSun']  # 指定默认字体
plt.rcParams['font.family'] = 'sans-serif'    # 无衬线字体（中文）
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

# 设置英文和数字字体为 Times New Roman
plt.rcParams['mathtext.fontset'] = 'stix'    # 使用 STIX 字体（支持 Times 类型）
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

# 生成唯一的路径，包含模型名称和时间戳
current_time = datetime.now().strftime("%y%m%d%H%M%S")
total_path = f".\\outputs\\GAC\\{current_time}\\Total\\"
os.makedirs(total_path, exist_ok=True)
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


# 读取并准备数据
data = pd.read_excel('.\\src\\swl_lstm\\data_GAC.xlsx')
origin_features = ['GA_Time', 'PH', 'NTU', 'UV254', 'TOC', 'Conductivity', 'ATZ', 'DEA', 'DIA','TMP', 'CBD', '2-AB', 'BZ']
target =['UV254', 'TOC', 'ATZ', 'DEA', 'DIA','TMP', 'CBD', '2-AB', 'BZ']
filtered_features = ['GA_Time', 'PH', 'UV254', 'TOC', 'ATZ', 'DEA', 'DIA','TMP', 'CBD', '2-AB', 'BZ']
#BZ: 14.65679，2-AB：8.7397，CBD：26.3088，TMP：25.09，DIA：3.27271，DEA：8.55317，ATZ：11.587.
toxicity=[0,0,11.587,25.09,26.3088,8.7397,14.65679,8.55317,3.27271]
toxicity=[x*0.01 for x in toxicity]
zh_target={feature: feature for feature in origin_features}
zh_target.update({'GA_Time':'活性炭时间','Conductivity':'电导率'})
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
time_step = 2

#存放三个模型9种物质的预测值与真实值
All_Trues=[]
CNN_LSTM_Preds=[]
LSTM_Preds=[]
Transformer_Preds=[]
#毒性
toxicity_Trues=[]
toxicity_CNN_LSTM_Preds=[]
toxicity_LSTM_Preds=[]
toxicity_Transformer_Preds=[]


# 创建一个函数来生成SHAP图
def generate_shap_plots(model, model_name, X_test, y_test):
    # 拆分MultiOutputRegressor的子模型
    sub_models = model.estimators_
    assert len(sub_models) == len(target), "子模型数量与目标变量数量不匹配"
    
    for output_idx in range(len(target)):
        target_name = target[output_idx]
        zh_target_name = zh_target.get(target_name, target_name)
        
        # 提取当前输出变量的子模型
        current_model = sub_models[output_idx]
        
        # 计算SHAP值
        explainer = shap.TreeExplainer(current_model)
        current_shap = explainer.shap_values(X_test)  # 形状 (n_samples, n_features)
        
        # 计算特征重要性
        feature_importance = np.mean(np.abs(current_shap), axis=0)
        sorted_idx = np.argsort(-feature_importance)
        
        # 绘制图表
        plt.figure(figsize=(5, 5))
        plt.barh(
            [origin_features[i] for i in sorted_idx],  # 使用过滤后的特征名称
            feature_importance[sorted_idx],
            color='#ff7f0e'
        )
        #plt.title(f"SHAP Feature Importance for {zh_target_name}", fontproperties=zh_font)
        #plt.xlabel("SHAP Value Magnitude")
        ax = plt.gca()
        ax.invert_yaxis()
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))  # x轴最多5个刻度
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        # 保存图表
        save_path = os.path.join(total_path, f"shap_{model_name}_{zh_target_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

for model_flag in [1, 2, 3]:
    FLAG = model_flag
    if FLAG == 1:
        model_name = "CNN_LSTM"
    elif FLAG == 2:
        model_name = "LSTM"
    elif FLAG == 3:
        model_name = "Transformer"

    
    path = f".\\outputs\\GAC\\{current_time}\\{model_name}\\"
    model_path = f"{path}{model_name}.h5"
    os.makedirs(path, exist_ok=True)
    res=""
    
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
    features=['GA_Time', 'PH', 'UV254', 'TOC', 'ATZ', 'DEA', 'DIA','TMP', 'CBD', '2-AB', 'BZ']
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

    #用于计算SHAP的预测
    x_test_shap, y_test_shap = [],[]

    for train_index, test_index in kf.split(X):
        # 划分训练集和测试集
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if(x_test_shap==[]):
            x_test_shap = X_test.copy()
            y_test_shap = y_test.copy()
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
        
        # 训练模型,500epoch
        history = model.fit(X_train, y_train, epochs=500, batch_size=32, validation_data=(X_test, y_test), verbose=0)
        
        model.save(model_path)
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
    #将列表转换为 numpy 数组
    all_preds = np.array(all_preds)
    # 替换所有负值为 0（逐元素操作）
    all_preds = np.where(all_preds < 0, 0, all_preds)# np.array(all_preds)#135,9 替换所有负值为0
    all_true = np.array(all_true)#135,9

    if FLAG == 1:
        CNN_LSTM_Preds = all_preds.copy()  # 直接存储 numpy 数组
    elif FLAG == 2:
        LSTM_Preds = all_preds.copy()
    elif FLAG == 3:
        Transformer_Preds = all_preds.copy()
    if All_Trues==[]:  # 首次赋值时存储真实值
        All_Trues = all_true.copy()

    # 替换各个时间段0点的预测值为真实值（这个应是已知条件）
    time_length=6
    for j in range(0,97-time_length,time_length):
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
    preds_df.to_csv(f"{path}predictions_and_true_values_{model_name}.csv", encoding='utf-8-sig')
    #res+=f'真实值:形状{all_true.shape}\n+{all_true}\n预测值:形状{all_preds.shape}\n+{all_preds}\n'

    # 绘制单个模型所有目标变量的预测对比图
    for i, target_name in enumerate(target):
        for j in range(0,97-time_length,time_length):
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
    # 计算毒性
    toxicity_predicted = np.sum(all_preds[:, 2:9] * toxicity[2:9], axis=1)
    toxicity_true = np.sum(all_true[:, 2:9] * toxicity[2:9], axis=1)
    if FLAG == 1:
        toxicity_CNN_LSTM_Preds = toxicity_predicted.copy()
    elif FLAG == 2:
        toxicity_LSTM_Preds = toxicity_predicted.copy()
    elif FLAG == 3:
        toxicity_Transformer_Preds = toxicity_predicted.copy()
    if toxicity_Trues==[]:  # 首次存储真实毒性值
        toxicity_Trues = toxicity_true.copy()
    with open(path+"\\output.txt", "a", encoding="utf-8") as file:
        file.write(res + "\n")

    #绘制SHAP
    X, y = create_dataset(origin_scaled_data_df, time_step)
    X_2d = X.reshape(X.shape[0], -1)  # 转换为 (n_samples, n_timesteps * n_features)
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_2d, y, test_size=0.2, random_state=42)
    # 训练多输出随机森林模型
    model = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=100, random_state=42)
    )
    model.fit(X_train, y_train)
    
    # 生成SHAP图
    generate_shap_plots(
        model=model,
        model_name=model_name,
        X_test=X_test,  # 使用测试集数据
        y_test=y_test
    )
     # 生成SHAP图
    
#保存到excel中，便于后续处理使用
# 1. 将物质数据保存为DataFrame
# 假设 All_Trues 是 (样本数, 9) 的二维数组
df_all_trues = pd.DataFrame(All_Trues, columns=target)  # target是物质名称列表

# 各模型的预测值
df_cnn_lstm = pd.DataFrame(CNN_LSTM_Preds, columns=target)
df_lstm = pd.DataFrame(LSTM_Preds, columns=target)
df_transformer = pd.DataFrame(Transformer_Preds, columns=target)

# 2. 将毒性数据保存为单独的DataFrame
df_toxicity_trues = pd.DataFrame({'毒性真实值': toxicity_Trues})
df_toxicity_cnn_lstm = pd.DataFrame({'毒性CNN_LSTM预测': toxicity_CNN_LSTM_Preds})
df_toxicity_lstm = pd.DataFrame({'毒性LSTM预测': toxicity_LSTM_Preds})
df_toxicity_transformer = pd.DataFrame({'毒性Transformer预测': toxicity_Transformer_Preds})

# 3. 合并所有DataFrame到Excel文件
with pd.ExcelWriter(f"{total_path}/model_results.xlsx") as writer:
    # 物质数据
    df_all_trues.to_excel(writer, sheet_name='9种物质真实值', index=False)
    df_cnn_lstm.to_excel(writer, sheet_name='CNN_LSTM物质预测', index=False)
    df_lstm.to_excel(writer, sheet_name='LSTM物质预测', index=False)
    df_transformer.to_excel(writer, sheet_name='Transformer物质预测', index=False)
    
    # 毒性数据
    df_toxicity_trues.to_excel(writer, sheet_name='毒性真实值', index=False)
    df_toxicity_cnn_lstm.to_excel(writer, sheet_name='CNN_LSTM毒性预测', index=False)
    df_toxicity_lstm.to_excel(writer, sheet_name='LSTM毒性预测', index=False)
    df_toxicity_transformer.to_excel(writer, sheet_name='Transformer毒性预测', index=False)
#最后总的绘图
# print(All_Trues)
# print(toxicity_Trues)

# print(CNN_LSTM_Preds)
# print(LSTM_Preds)
# print(Transformer_Preds)

# print(toxicity_CNN_LSTM_Preds)
# print(toxicity_LSTM_Preds)
# print(toxicity_Transformer_Preds)

# 绘制毒性对比图
plt.figure(figsize=(21, 3))
plt.plot(toxicity_Trues, label='真实值', color='black')
plt.plot(toxicity_CNN_LSTM_Preds, label='CNN_LSTM预测值', color='red')
plt.plot(toxicity_LSTM_Preds, label='LSTM预测值', color='blue')
plt.plot(toxicity_Transformer_Preds, label='Transformer预测值', color='green')
# 设置坐标轴标签和标题
#plt.xlabel('时间步')
#plt.ylabel('综合毒性值')
#plt.title('综合毒性预测对比')
# 添加垂直虚线（每隔8个单位）
max_x = len(toxicity_Trues)  # 获取x轴最大值
# 计算所有垂直线的位置
positions = list(range(0, max_x + 1, 6))  # 生成0,9,18,...等位置
# 绘制垂直虚线
for x in positions:
    plt.axvline(x, color='gray', linestyle='--', alpha=0.5)
# 设置x轴刻度仅显示这些位置，并设置字体大小
plt.xticks(ticks=positions, fontsize=20)  # 显式指定刻度位置
# 设置y轴刻度字体大小（保持原有逻辑）
plt.yticks(fontsize=20)
#plt.legend() 
plt.savefig(f"{total_path}/toxicity_comparison.png", dpi=300)
plt.close()

# 绘制三个模型所有目标变量的预测对比图
for i, target_name in enumerate(target):
    plt.figure(figsize=(21, 3))
    #plt.plot(All_Trues[:, i], label=f'真实值 {zh_target.get(target_name)}')
    # 真实值
    plt.plot(All_Trues[:, i], label='真实值', color='black')
    # 三个模型预测值
    plt.plot(CNN_LSTM_Preds[:, i], label='CNN_LSTM预测值', color='red')
    plt.plot(LSTM_Preds[:, i], label='LSTM预测值', color='blue')
    plt.plot(Transformer_Preds[:, i], label='Transformer预测值', color='green')
    # 添加垂直虚线（每隔8个单位）
    max_x = All_Trues.shape[0]  # 获取x轴最大值
    # 计算所有垂直线的位置
    positions = list(range(0, max_x + 1, 6))  # 生成0,9,18,...等位置
    # 绘制垂直虚线
    for x in positions:
        plt.axvline(x, color='gray', linestyle='--', alpha=0.5)
    # 设置x轴刻度仅显示这些位置，并设置字体大小
    plt.xticks(ticks=positions, fontsize=20)  # 显式指定刻度位置
    plt.yticks(fontsize=20)
    #plt.title(f'{zh_target.get(target_name)} 预测值和真实值对比图')
    #plt.xlabel('臭氧反应时间(分钟)')
    #plt.ylabel(f'{zh_target.get(target_name)}浓度(ppb)')
    # 添加图例，并去掉边框
    #plt.legend()
    plt.savefig(f"{total_path}/{target_name}_comparison.png", dpi=300)
    plt.close()

#绘制(9种物质+1种毒性)*3种模型的散点图
#横轴为真实值，纵轴为预测值，画一条对角线，不需要图例和xlabe、ylabel、title。
# 遍历所有目标变量和模型
# 定义模型字典（假设模型预测值已存在）
models = {
    'CNN_LSTM': CNN_LSTM_Preds,
    'LSTM': LSTM_Preds,
    'Transformer': Transformer_Preds
}
#9种物质
for i, target_name in enumerate(target):
    true_values = All_Trues[:, i]
    # 获取中文名称（假设zh_target已定义）
    #cn_target_name = zh_target.get(target_name, target_name)
    
    for model_name, preds in models.items():
        pred_values = preds[:, i]
        
        # 计算分位数（过滤最大20%的点）
        true_80_percentile = np.percentile(true_values, 90)
        pred_80_percentile = np.percentile(pred_values, 90)
        
        # 创建过滤掩码：保留 <= 80分位数的点（即过滤掉最大的20%）
        mask = (true_values <= true_80_percentile) & (pred_values <= pred_80_percentile)
        
        # 过滤后的数据
        filtered_true = true_values[mask]
        filtered_pred = pred_values[mask]
        
        # 创建画布
        plt.figure(figsize=(4, 4))
        
        # 绘制散点图（仅过滤后的数据）
        plt.scatter(filtered_true, filtered_pred, alpha=0.6,color='green', edgecolors='w', s=40)
        
        # 计算坐标轴范围
        max_val = max(max(filtered_true), max(filtered_pred)) * 1.05  # 扩展5%边距
        min_val = min(min(filtered_true), min(filtered_pred)) * 0.95
        
        # 绘制对角线（理想预测线）
        plt.plot([min_val, max_val], [min_val, max_val], 
                 color='blue', linestyle='--', linewidth=1)
        
        # 设置坐标轴范围
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)

        # **新增代码：统一刻度间隔**
        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=3))  # x轴最多4个刻度
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=3))  # y轴最多4个刻度
        # 隐藏坐标轴标签（保留刻度数字）
        plt.xlabel('')
        plt.ylabel('')
        
        # 可选：隐藏坐标轴刻度标签（如果需要完全隐藏）
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        
        # 保存图片（使用中文名称命名）
        filename = f"scatter_{target_name}_{model_name}_filtered.png"
        plt.savefig(f"{total_path}/{filename}", dpi=300, bbox_inches='tight')
        plt.close()

#毒性
# 定义毒性模型数据
toxicity_models = {
    'CNN_LSTM': toxicity_CNN_LSTM_Preds,
    'LSTM': toxicity_LSTM_Preds,
    'Transformer': toxicity_Transformer_Preds
}
# 综合毒性的目标名称（假设为 "综合毒性"）
target_name_toxicity = "综合毒性"  # 可根据实际名称调整

# 遍历每个模型绘制散点图
for model_name, pred_values in toxicity_models.items():
    true_values = toxicity_Trues  # 真实毒性值
    # 计算分位数（过滤最大20%的点）
    true_80_percentile = np.percentile(true_values, 80)  # 注意：此处分位数需与之前一致
    pred_80_percentile = np.percentile(pred_values, 80)
    # 创建过滤掩码：保留 <= 90分位数的点（即过滤掉最大的10%）
    mask = (true_values <= true_80_percentile) & (pred_values <= pred_80_percentile)
    # 过滤后的数据
    filtered_true = true_values[mask]
    filtered_pred = pred_values[mask]
    # 创建画布
    plt.figure(figsize=(4, 4))
    
    # 绘制散点图（仅过滤后的数据）
    plt.scatter(filtered_true, filtered_pred, 
                alpha=0.6, edgecolors='w', s=40)
    # 计算坐标轴范围
    max_val = max(max(filtered_true), max(filtered_pred)) * 1.05  # 扩展5%边距
    min_val = min(min(filtered_true), min(filtered_pred)) * 0.95
    # 绘制对角线（理想预测线）
    plt.plot([min_val, max_val], [min_val, max_val], 
             color='red', linestyle='--', linewidth=1)
    # 设置坐标轴范围
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    # **新增代码：统一刻度间隔**
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=3))  # x轴最多4个刻度
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=3))  # y轴最多4个刻度
    # 隐藏坐标轴标签（保留刻度数字）
    plt.xlabel('')
    plt.ylabel('')
    # 设置刻度字体大小（与之前一致）
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # 保存图片（使用中文名称命名）
    filename = f"scatter_{target_name_toxicity}_{model_name}_filtered.png"
    plt.savefig(f"{total_path}/{filename}", dpi=300, bbox_inches='tight')
    plt.close()

# 计算SHAP



    
