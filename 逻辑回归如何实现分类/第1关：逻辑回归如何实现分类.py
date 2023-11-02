from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
import numpy as np
np.random.seed(10)
if __name__ == '__main__':
    # 请在下面构建二元分类数据,样本数量为100,并将分类数据中的80%作为测试集
    # 对二分类数据进行分类,并评估模型的准确率
    # 代码中设置了随机种子,请按照要求设计程序，否则可能导致程序不通过
    # ********** Begin ********** #
    # 使用sklearn中的make_classification函数构建二分类的数据
    x, y = make_classification(n_samples=100, n_classes=2)
    # 将数据集拆分成测试集与训练集，训练集占所有数据的80%
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    # 进行模型训练兵评估模型准确率
    logistic = LogisticRegressionCV()
    logistic.fit(x_train, y_train)
    score = logistic.score(x_test, y_test)
    print(score)
    # ********** End ********** #