#encoding=utf8
import numpy as np

class kNNClassifier(object):
    def __init__(self, k):
        '''
        初始化函数
        :param k:kNN算法中的k
        '''
        self.k = k
        # 用来存放训练数据，类型为ndarray
        self.train_feature = None
        # 用来存放训练标签，类型为ndarray
        self.train_label = None


    def fit(self, feature, label):
        '''
        kNN算法的训练过程
        :param feature: 训练集数据，类型为ndarray
        :param label: 训练集标签，类型为ndarray
        :return: 无返回
        '''

        #********* Begin *********#
        self.train_feature = np.array(feature)
        self.train_label = np.array(label)
        #********* End *********#


    def predict(self, feature):
        '''
        kNN算法的预测过程
        :param feature: 测试集数据，类型为ndarray
        :return: 预测结果，类型为ndarray或list
        '''

        #********* Begin *********#
        def _predict(test_data):
            distances = [np.sqrt(np.sum((test_data - vec) ** 2)) for vec in self.train_feature]
            nearest = np.argsort(distances)
            topK = [self.train_label[i] for i in nearest[:self.k]]
            votes = {}
            result = None
            max_count = 0
            for label in topK:
                if label in votes.keys():
                    votes[label] += 1
                    if votes[label] > max_count:
                        max_count = votes[label]
                        result = label
                else:
                    votes[label] = 1
                    if votes[label] > max_count:
                        max_count = votes[label]
                        result = label
            return result
        predict_result = [_predict(test_data) for test_data in feature]
        return predict_result

        #********* End *********#
