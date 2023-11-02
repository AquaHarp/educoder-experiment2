import numpy as np
 
class Loss(object):
 
    def mean_absolute_loss(self,y_hat,y,n):
        """
        平均绝对误差损失
        :param y_hat: 预测结果
        :param y: 真实结果
        :param n: 样本数量
        :return: 损失函数计算结果
        """
        ########## Begin ##########
        loss = abs(y_hat - y)
        loss = np.sum(loss) / n
        ########## End ##########
        return loss
 
    def mean_squared_loss(self,y_hat,y,n):
        """
        均方差损失
        :param y_hat: 预测结果
        :param y: 真实结果
        :param n: 样本数量
        :return: 损失函数计算结果
        """
        ########## Begin ##########
        loss = (y_hat - y) ** 2
        loss = np.sum(loss) / n
        ########## End ##########
        return loss
 
    def cross_entropy_loss(self,y_hat,y,n):
        """
        交叉熵损失
        :param y_hat: 预测结果
        :param y: 真实结果
        :param n: 样本个数
        :return: 损失函数计算结果
        """
        ########## Begin ##########
        # loss = -np.sum(y*np.log(y_hat)) + (1-y)*np.log(1-y_hat)
        loss = y_hat * np.log2(y) + (1 - y) * np.log2(1 - y)
        loss = np.sum(loss)
        loss = loss / (-n)
        ########## End ##########
        return loss
 
 