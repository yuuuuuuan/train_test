import numpy as np
import pandas as pd

# 设置随机种子以保证结果可复现
np.random.seed(42)

# 假设我们需要 1000 个样本，4 个特征
n_samples = 1000
n_features = 4

# 生成随机特征数据
X = np.random.randn(n_samples, n_features)

# 为了保持线性关系，假设目标变量 y 是 X 的线性组合，再加上一些噪声
true_coefficients = np.array([2.5, -1.5, 3.0, 0.5])  # 每个特征的真实系数
noise = np.random.randn(n_samples) * 5  # 噪声项

# 计算目标变量 y
y = X.dot(true_coefficients) + noise

# 创建 DataFrame
df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(n_features)])
df['target'] = y

# 将数据保存为 CSV 文件
df.to_csv('linear_data.csv', index=False)

print("CSV 文件 'linear_data.csv' 已生成。")
