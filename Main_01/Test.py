import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from A import LinearRegression


data = pd.read_csv('world-happiness-report-2017.csv')

#测试数据集合和训练数据集合
train_data = data.sample(frac=0.8)#用数据的80%训练
test_data =data.drop(train_data.index)#.drop是删除指定的列，用剩下的数据测试

#输入特征和输出特征
input_param_name = 'Economy..GDP.per.Capita.' #X
output_param_name = 'Happiness.Score' #Y

x_train = train_data[[input_param_name]].values
y_train = train_data[[output_param_name]].values

x_test = test_data[[input_param_name]].values
y_test=test_data[[output_param_name]].values

#数据展示
plt.scatter(x_train,y_train,label='Train data')
plt.scatter(x_test,y_test,label='Test data')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title("Test")
plt.legend()#图例是一种用于标识图中不同元素的说明性标记。图例通常用于区分不同的数据系列、不同的曲线、不同的颜色或样式等
plt.show()

#Train

#rations
num_iterations=500
#Learning rate
learning_rate=0.01

linear_regression=LinearRegression(x_train,y_train)

(theta,cost_history)=linear_regression.train(learning_rate,num_iterations)

print('开始时候损失：',cost_history[0])
print('训练后的：',cost_history[-1])