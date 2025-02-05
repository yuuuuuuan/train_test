import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# 假设data是加载的原始数据
data = pd.read_csv('housing_data.csv')

# 数据预处理
# 将日期转化为日期类型并提取年份
data['date'] = pd.to_datetime(data['date'])
data['year_built'] = data['date'].dt.year - data['yr_built']
data.drop(['date', 'yr_built'], axis=1, inplace=True)

# 定义特征与标签
X = data.drop('price', axis=1)
y = data['price']

# 分类变量处理
#categorical_features = ['street', 'city', 'statezip']
numeric_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement', 'yr_renovated']

# 创建预处理器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numeric_features),
        #('cat', OneHotEncoder(), categorical_features)
    ])

# 创建管道
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 模型训练
model.fit(X_train, y_train)

# 预测与评估
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
