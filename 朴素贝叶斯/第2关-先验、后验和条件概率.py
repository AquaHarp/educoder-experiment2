# 导入库
import numpy as np

# 共 100 个样本，每个样本 x 都包括 5 个特征
np.random.seed(0)
x = np.random.randn(100, 5)

# 共 100 个样本，每个样本 x 都属于 {0,1,2，...，9} 类别中的一个
np.random.seed(0)
y = np.random.randint(0,10,100)

# 初始化先验概率，P[i] 表示类别 i 出现的概率
P = np.zeros(100)

# 任务1：计算每个标签的先验概率
########## Begin ##########
for i in range(10):
    P[i] = np.sum(np.where(y==i, 1, 0))/100
##########  End  ##########

# 打印结果
for i in range(10):
    print("类别 i 出现的概率为：", P[i])
