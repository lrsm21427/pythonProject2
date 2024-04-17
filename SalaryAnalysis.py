import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 导入数据
data = pd.read_csv("D:\MachineLearning\pythonProject2\Salary_dataset.csv")

# var = data.columns
# print("表头：\n",var)
#
# var = data.shape
# print("维度信息：\n",var)
#
# var = data.head()
# print("前5行：\n",var)
#
# var = data.describe()
# print("统计摘要：\n",var)
#
# var = data.info()
# print("非空值数量和数据类型：\n",var)
#
# var = data.isnull().sum()
# print("每列中的缺失值数量：\n",var)
#
# data.drop(['Unnamed: 0'], axis=1, inplace=True)

##绘制图像查看数据
# # 薪资和经验散点图
# sns.scatterplot(data=data, x="YearsExperience", y="Salary")
# plt.show()
#
# #各个数值列之间的成对关系
# sns.pairplot(data)
# plt.show()
#
# #经验的箱线图
# sns.boxplot(data["YearsExperience"])
# plt.show()
#
# #薪资的箱线图
# sns.boxplot(data=data["Salary"])
# plt.show()
#
# #绘制经验变量的直方图和核密度估计图
# sns.displot(data["YearsExperience"], kde=True)
# plt.show()
#
# #绘制薪资变量的直方图和核密度估计图
# sns.displot(data["Salary"], kde=True)
# plt.show()

#创建DataFrame的副本以保持原始数据的完整性
data_scaled = data.copy()

#特征缩放：最小-最大标量
min_max = MinMaxScaler()

#对工作经验（YearsExperience）进行缩放
min_max_yoe = min_max.fit_transform(data_scaled[["YearsExperience"]])
data["ScaledYearsExperience"] = min_max_yoe

# 提取特征和目标变量
x = data['YearsExperience'].values.reshape(-1, 1)  # 特征
y = data['Salary'].values  # 目标变量

# # 将数据集分割为训练集和测试集（保持一致）
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#创建线性回归模型对象
linear_reg = LinearRegression()
linear_reg.fit(x_train, y_train)

#线性回归模型的斜率
slope = linear_reg.coef_
print("斜率（Slope）:", slope)

#模型的截距
intercept = linear_reg.intercept_
print("截距（Y-Intercept）:", intercept)

#对测试集的特征变量进行预测
predictions = linear_reg.predict(x_test)
print("预测的测试值:", predictions)

#对给定的工作经验值进行预测
random_prediction = linear_reg.predict([[10]])
print("对于工作经验值为10的线性回归预测薪资:", random_prediction)


#线性回归模型在测试集上的R²分数
r2 = r2_score(y_test, linear_reg.predict(x_test))
print("R²分数:", r2)

# print("训练集特征（x_train）:\n", x_train)
# print("训练集目标变量（y_train）:\n", y_train)
# print("测试集特征（x_test）:\n", x_test)
# print("测试集目标变量（y_test）:\n", y_test)


# 创建随机森林回归模型对象
random_forest_reg = RandomForestRegressor()

# 在训练集上拟合模型
random_forest_reg.fit(x_train, y_train)

# 对测试集的特征变量进行预测
predictions_rf = random_forest_reg.predict(x_test)
print("随机森林预测的测试值:", predictions_rf)

# 对给定的工作经验值进行预测
random_prediction_rf = random_forest_reg.predict([[10]])
print("对于工作经验值为10的随机森林预测薪资:", random_prediction_rf)

# 随机森林模型在测试集上的R²分数
r2_rf = r2_score(y_test, predictions_rf)
print("随机森林模型的R²分数:", r2_rf)


# # 计算训练集上的均方误差和平均绝对误差
# train_predictions = linear_reg.predict(x_train)
# mse_train = mean_squared_error(y_train, train_predictions)
# mae_train = mean_absolute_error(y_train, train_predictions)
# print("训练集均方误差（MSE）:", mse_train)
# print("训练集平均绝对误差（MAE）:", mae_train)
#
# # 计算测试集上的均方误差和平均绝对误差
# mse_test = mean_squared_error(y_test, predictions)
# mae_test = mean_absolute_error(y_test, predictions)
# print("测试集均方误差（MSE）:", mse_test)
# print("测试集平均绝对误差（MAE）:", mae_test)
#
# # 随机森林模型的均方误差和平均绝对误差
# mse_rf = mean_squared_error(y_test, predictions_rf)
# mae_rf = mean_absolute_error(y_test, predictions_rf)
# print("随机森林模型的测试集均方误差（MSE）:", mse_rf)
# print("随机森林模型的测试集平均绝对误差（MAE）:", mae_rf)
