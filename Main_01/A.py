import numpy as np
from utils.features.prepare_for_training import prepare_for_training
class LinearRegression:
    # 预处理
    def __init__(self,data,labels,polynoimal_degree=0,sinusoid_degree=0,normalize_data=True):
            (data_processed,
             features_mean,
             features_deviation)=prepare_for_training(data,polynomial_degree=0, sinusoid_degree=0, normalize_data=True)
            self.data=data_processed
            self.labels=labels
            self.features_mean=features_mean
            self.features_deviation=features_deviation
            #变换
            self.polynoimal_degree=polynoimal_degree
            self.sinusoid_degree=sinusoid_degree
            self.normalize_data=normalize_data

            #theta（θ） 参数：应该和特征参数一一对应
            num_features =self.data.shape[1]#列
            self.theta = np.zeros((num_features,1))#初始化
    # 训练
    def train(self,alpha,num_iterations=500):#学习率和迭代次数
        cost_history = self.gradient_descent(alpha,num_iterations) #Alpha 和 迭代次数
        return self.theta,cost_history

    # 执行一次梯度下降
    # 参数更新
    def gradient_descent(self,alpha,num_iterations):
        cost_history =[]
        for i in range(num_iterations): #每一次迭代
            self.gradient_step(alpha)
            cost_history.append(self.cost_function(self.data,self.labels))#计算损失
        return cost_history

    def gradient_step(self,alpha):#计算步骤
        # 梯度下降参数更新方法，矩阵运算
        num_examples=self.data.shape[0] #样本的个数
        prediction =LinearRegression.hypothesis(self.data,self.theta)#当前数据和当前theta
        #预测值减真实值
        delta = prediction - self.labels
        theta=self.theta
        #更新theta（公式）
            #计算
        theta=theta-alpha*(1/num_examples)*(np.dot(delta.T,self.data)).T
            #更新
        self.theta=theta
     # 损失函数
    def cost_function(self, data, labels):
      num_examples =data.shape[0]#总个数
      delta=LinearRegression.hypothesis(self.data,self.theta)-labels #预测值-labels
      #损失值
      cost = (1/2)*np.dot(delta.T,delta)
      return cost [0][0]
    @staticmethod #装饰器用于定义静态方法。静态方法是属于类的方法，与实例对象无关，可以直接通过类名调用，而无需创建类的实例。
    def hypothesis(data,theta):#预测函数
        prediction=np.dot(data,theta)#真实值 乘以 当前的参数（θ）
        return prediction
    #完善
    #1得到当前的损失
    def get_cost(self,data,labels):
        data_processed=prepare_for_training(data,
                             self.polynoimal_degree,
                             self.sinusoid_degree,
                             self.normalize_data
                             )[0]
        return self.cost_function(data_processed,labels)
    def predict(self,data):#用训练好的参数模型去预测回归值结果
        data_processed=prepare_for_training(data,
                             self.polynoimal_degree,
                             self.sinusoid_degree,
                             self.normalize_data
                             )[0]
        predictions =LinearRegression.hypothesis(data_processed,self.theta)
        return predictions