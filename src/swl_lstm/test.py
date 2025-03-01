import pandas as pd
import numpy as np

# 示例数据
data = pd.DataFrame({
    'O3': [0, 0.25, 0.5, 1, 2, 0, 0.25, 0.5, 1, 2, 0, 0.25, 0.5, 1, 2, 0, 0.25, 0.5, 1, 2,
           0, 0.25, 0.5, 1, 2, 3, 5, 10, 15, 0, 0.25, 0.5, 1, 2, 3, 5, 10, 15, 0, 2.5, 5, 10,
           15, 20, 30, 45, 60, 80, 100, 120, 10, 10, 10, 10, 10, 15, 15, 15, 15, 15, 20, 20, 20,
           20, 20, 30, 30, 30, 30, 30, 0, 5, 10, 15, 20, 25, 30, 35, 40, 5, 5, 5, 5, 5, 10, 10, 10,
           10, 10, 15, 15, 15, 15, 15, 20, 20, 20, 20, 20, 25, 25, 25, 25, 25, 30, 30, 30, 30, 30,
           35, 35, 35, 35, 35, 40, 40, 40, 40, 40],
    'GA': [0, 0, 0, 0, 0, 20, 20, 20, 20, 20, 0, 0, 0, 0, 0, 20, 20, 20, 20, 20, 0, 0, 0, 0, 0,
           0, 20, 20, 20, 20, 0, 20, 20, 20, 20, 20, 20, 20, 20, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           5, 10, 15, 20, 0, 5, 10, 15, 20, 0, 5, 10, 15, 20, 20, 5, 10, 15, 20, 0, 5, 10, 15, 20, 25,
           0, 5, 10, 15, 20, 25, 25, 30, 35, 40, 0, 5, 10, 15, 20, 25, 30, 35, 40]
})

def detect_sequences(data, threshold=0.5):
    sequences = []
    current_sequence = []
    start_idx = 0
    while start_idx < len(data):
        end_idx = start_idx
        current_sequence = [data.iloc[start_idx]]
        while end_idx + 1 < len(data) and (
            abs(data['O3'].iloc[end_idx + 1] - data['O3'].iloc[end_idx]) <= threshold and
            abs(data['GA'].iloc[end_idx + 1] - data['GA'].iloc[end_idx]) <= threshold):
            end_idx += 1
            current_sequence.append(data.iloc[end_idx])
        
        sequences.append(pd.DataFrame(current_sequence))
        start_idx = end_idx + 1
    
    return sequences

# 划分数据
split_data = detect_sequences(data)
print(split_data.type)