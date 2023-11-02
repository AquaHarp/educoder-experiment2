from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import matplotlib.pyplot as plt

def digit_predict(train_image, train_label, test_image):
    '''
    实现功能：训练模型并输出预测结果
    :param train_sample: 包含多条训练样本的样本集，类型为ndarray,shape为[-1, 8, 8]
    :param train_label: 包含多条训练样本标签的标签集，类型为ndarray
    :param test_sample: 包含多条测试样本的测试集，类型为ndarry
    :return: test_sample对应的预测标签
    '''

    #************* Begin ************#
    # 训练集变形
    flat_train_image = train_image.reshape((-1, 64))
    # 训练集标准化
    train_min = flat_train_image.min()
    train_max = flat_train_image.max()
    flat_train_image = (flat_train_image-train_min)/(train_max-train_min)
    # 测试集变形
    flat_test_image = test_image.reshape((-1, 64))
    # 测试集标准化
    test_min = flat_test_image.min()
    test_max = flat_test_image.max()
    flat_test_image = (flat_test_image - test_min) / (test_max - test_min)

    # 训练--预测
    rf = LogisticRegression(C=4.0)
    rf.fit(flat_train_image, train_label)
    return rf.predict(flat_test_image)

    #************* End **************#