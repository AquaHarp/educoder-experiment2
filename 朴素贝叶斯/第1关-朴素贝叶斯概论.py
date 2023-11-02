# 导入库
import numpy as np

# 共 100 个样本，每个样本 x 都包括 5 个特征
np.random.seed(0)
x = np.random.randn(100, 5)

# 共 100 个样本，每个样本 x 都属于 {0,1} 类别中的一个
np.random.seed(0)
y = np.random.randint(0,2,100)

# 给定 xx = [0,1,0,1,1]
xx = np.array([0,1,0,1,1])

setx = []
# setx[i] 表示属于类别 i 的 x 的集合
for i in range(2):
    setx.append(x[np.where(y==i)])

p = []
# 初始化 p，p[i] 表示 xx 属于类别 i 的概率
for i in range(2):
    p.append(setx[i].shape[0] / 100)

# 正态分布的概率密度函数
def normal(x, mean, std):
    return np.exp(-(x-mean)**2/(2*std**2))/(np.sqrt(2*np.pi)*std)

# 任务1：根据条件独立假设，求样本 xx 属于 i 类别的概率
########## Begin ##########
for i in range(2):
    for j in range(5):
        mean = np.mean(setx[i][:, j])
        std = np.std(setx[i][:, j])
        p[i] = p[i] * normal(xx[j], mean, std)
##########  End  ##########

# 任务2：根据连续型朴素贝叶斯判定准则，求样本 xx 属于哪个类别的概率最大
########## Begin ##########
label = np.argmax(p)
##########  End  ##########

# 打印结果
print("样本 xx = [0,1,0,1,1] 属于类别", label)
