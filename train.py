import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 加载生成的CSV数据
df = pd.read_csv('linear_data.csv')

# 特征和目标变量
X = df.drop(columns=['target'])  # 特征
y = df['target']  # 目标变量

# 将数据集分为训练集和测试集，80% 用于训练，20% 用于测试
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化线性回归模型
model = LinearRegression()

# 训练模型
model.fit(X_train, y_train)

# 用训练好的模型在测试集上进行预测
y_pred = model.predict(X_test)

# 计算模型的均方误差
mse = mean_squared_error(y_test, y_pred)

# 输出训练结果和均方误差
print("训练完成！")
print(f"均方误差 (MSE): {mse}")
print("预测的目标值 (target):")
print(y_pred[:10])  # 打印前 10 个预测值

# 保存训练好的模型
joblib.dump(model, 'linear_regression_model.pkl')

# 你可以随时加载模型并进行推理
loaded_model = joblib.load('linear_regression_model.pkl')

# 用加载的模型进行预测
y_pred_loaded = loaded_model.predict(X_test)

# 打印加载模型后的预测结果
print("加载模型后的预测结果:")
print(y_pred_loaded[:10])

# 你也可以检查加载后的模型的性能
mse_loaded = mean_squared_error(y_test, y_pred_loaded)
print(f"加载模型后的均方误差 (MSE): {mse_loaded}")
