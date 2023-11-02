from sklearn.linear_model import LinearRegression

x = [[1], [4], [6], [10], [12]] 
y = [3, 5, 7, 10, 6]


# 1.建立模型并拟合
# 2.赋值给变量lr
#########  Begin #########

lr = LinearRegression().fit(x,y)  # 关键点
k = lr.coef_[0]
b = lr.intercept_
#########  end  ##########


# 1.得到回归模型的k值并赋值给变量k
# 2.得到回归模型的b值并赋值给变量b
#########  Begin #########


#########  end  ##########

print("回归直线方程为：" + "y = {:.3f}".format(k) + "x" + " + " + "{:.3f}".format(b))
