# 导入库
import numpy as np

# 共 100 个样本，每个样本 x 都包括 5 个特征
np.random.seed(0)
x = np.random.randint(0,2,(100, 5))

# 共 100 个样本，每个样本 x 都属于 {0,1} 类别中的一个
np.random.seed(0)
y = np.random.randint(0,2,100)

# 给定 xx = [0,1,0,1,1]
xx = np.array([0,1,0,1,1])

# setx_0 表示属于第一个类别的 x 的集合
setx_0 = x[np.where(y==0)]

# 初始化 p_0，p_0 表示 xx 属于类别 0 的概率
p_0 = setx_0.shape[0] / 100

# 任务1：根据条件独立假设，求样本 xx 属于第一个类别的概率
########## Begin ##########
for i in range(5):
    p_0 = p_0 * np.sum(np.where(setx_0[:, i]==xx[i], 1, 0)) / setx_0.shape[0]
##########  End  ##########

# 打印结果
print("样本 xx = [0,1,0,1,1] 属于类别 0 的概率为：", p_0)
