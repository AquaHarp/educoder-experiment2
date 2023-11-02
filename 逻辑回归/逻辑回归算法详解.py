from sklearn import datasets
from sklearn.datasets import load_iris 
import numpy as np
import math
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.linear_model import LogisticRegression   #导入逻辑回归模型

#########Begin########
# 导入数据
iris = datasets.load_iris()
X= iris['data']
y = iris['target']
X = X[y!=2] #  筛选数据，只选择标签为0和1
y=y[y!=2]
# 数据划分


# 模型调用
logr = LogisticRegression(solver='liblinear')
# 模型训练
logr.fit(X, y)
# 数据预测
X_test_pred = logr.predict(X)
# 结果打印
acc = logr.score(X,y)
print("准确度:",acc)
########End#########