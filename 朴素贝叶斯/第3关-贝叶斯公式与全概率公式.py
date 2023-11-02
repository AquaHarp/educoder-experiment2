# 导入库
import numpy as np

# 共 10 个盒子，b[i][0] 表示盒子 i 中的苹果数量，b[i][1] 表示盒子 i 中的橙子数量
np.random.seed(0)
b = np.random.randint(0,10,(10, 2))

# 共 10 个盒子，p[i] 表示盒子 i 被挑中的概率
p = np.array([0.1, 0.1, 0.05, 0.15, 0.08, 0.12, 0.09, 0.11, 0.06, 0.14])

# 初始化概率，P 表示挑出的水果是橙子的概率
P = 0

# 任务1：根据全概率公式，求挑出的水果是橙子的概率
########## Begin ##########
for i in range(10):
    P = P + p[i] * b[i][1] / (np.sum(b[i]))
##########  End  ##########

# 打印结果
print("挑出的水果是橙子的概率为：", P)

# 任务2：已知挑出的水果是橙子，根据贝叶斯公式，求是从第一个盒子挑出的概率
########## Begin ##########
P_1 = p[0] * b[0][1] / (np.sum(b[0])) / P
##########  End  ##########

# 打印结果
print("已知挑出的水果是橙子，是从第一个盒子挑出的概率为：", P_1)
