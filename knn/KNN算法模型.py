from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_blobs


da = make_blobs(n_samples=500, centers=5, random_state=8)
x, y = da

# 1.建立KNN算法模型
# 2.对数据进行拟合，并赋值给clf
#########  Begin #########
clf = KNeighborsClassifier().fit(x, y)


#########  end  ##########


print("模型评估：{:.2f}".format(clf.score(x, y)))
print(clf.predict([[1, 1]]))
print(clf.predict([[5, 5]]))
print(clf.predict([[10, 10]]))

