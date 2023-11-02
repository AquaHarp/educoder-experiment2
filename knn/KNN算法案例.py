import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine  # 载入酒的数据集模块


wine_dataset = load_wine()

# 1.将数据集划分成训练集和测试集，并且random_state设置为8
# 2.赋值给对应的变量：x_train, x_test, y_train, y_test
#########  Begin #########
X_train, X_test, y_train, y_test = train_test_split(wine_dataset.data, wine_dataset.target, test_size=0.31, random_state=8)

#########  end  ##########


# 1.建立KNN算法模型，K值为5
# 2.对训练集进行拟合，并赋值给clf
#########  Begin #########

clf = KNeighborsClassifier(n_neighbors=5) 
lr = clf.fit(X_train, y_train) 
#########  end  ##########



print("测试集正确率：{:.2f}".format(clf.score(X_test, y_test)))


da = np.array([[9.11, 21.1, 12.1, 11.3, 130.0, 2.2, 1.5, 0.1, 2.4, 3.0, 5.05, 1.3, 999]])

# 1.使用建立好的模型对新数据进行预测
# 2.赋值给predict_1
#########  Begin #########
predict_1 = clf.predict(da)

#########  end  ##########


print(wine_dataset["target_names"][predict_1])

