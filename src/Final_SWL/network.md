## CNN+LSTM

```python
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
```

```LSTM
# 构建LSTM模型
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))  # 添加Dropout防止过拟合
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(11))  # 输出11个值，对应预测的目标属性
```

Transformer

```python
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
    outputs = Dense(11)(ffn_output[:, -1, :])  # 只取最后一个时间步的输出
    model = Model(inputs, outputs)
    return model
```
