import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 导入数据
df = pd.read_csv("D:/JetBrainsProjects/pycharm/pythonProject2/Salary_dataset.csv")

df.drop(columns="Unnamed: 0",axis = 1,inplace = True)
#输出数据
print(df)
#null值
df.isnull().sum()

#数据可视化
plt.figure(figsize=(10,8))
sns.scatterplot(data=df,x="YearsExperience",y="Salary")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

#将数据集分离为相关特征和独立特征
x=df["YearsExperience"]
y=df["Salary"]
x.shape

#将自变量的值从一维重塑为二维
x = pd.DataFrame(x)
x.shape

#将数据集分为训练集和测试集
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 1)
#输出训练集和测试集的形状
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

#创建线性回归模型
model = LinearRegression()
model.fit(x_train,y_train)

#模型系数
print("斜率系数:", model.coef_)
print("截距:", model.intercept_)

#模型评估
plt.figure(figsize=(10,8))
plt.scatter(x_train,y_train)
plt.plot(x_train,model.predict(x_train),c="r")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

#模型评估
y_pred = model.predict(x_test)
print("预测值:", y_pred)
print("实际值:", y_test.values)

#可视化测试集上的实际值和预测值
plt.figure(figsize=(10, 8))
plt.scatter(x_test, y_test)
plt.plot(x_test,model.predict(x_test),c="r")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

#模型评估
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# 计算调整后的 R²
n = len(y_test)
k = x_test.shape[1]
adj_r2 = 1 - ((1 - r2) * (n - 1)) / (n - k - 1)

#性能指标
print("均方误差（MSE）:", mse)
print("平均绝对误差（MAE）:", mae)
print("R² 分数:", r2)
print("调整后的 R² 分数:", adj_r2)

#使用了 Ridge 回归模型来训练数据。Ridge 回归是一种用于处理回归问题的线性模型，它通过对系数施加惩罚来解决普通线性回归中的过拟合问题。这种惩罚可以帮助减小特征的影响，从而提高模型的泛化能力。
ridge = Ridge(alpha=0.01)
ridge.fit(x_train,y_train)

y_ridge_pred = ridge.predict(x_test)

# 计算调整后的 R²
ad_r2 = 1-((1-r2_score(y_test,y_ridge_pred))*(len(y_test)-1))/(len(y_test)-x_test.shape[1]-1)

#性能指标
print("均方误差（MSE）:" ,mean_squared_error(y_test,y_ridge_pred))
print('平均绝对误差（MAE）:' ,mean_absolute_error(y_test,y_ridge_pred))
print("R² 分数:", r2)
print("调整后的 R² 分数:", ad_r2)