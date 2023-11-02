#---------参考答案在此---------#
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.datasets import load_iris
def model_train(x_train, x_test, y_train, y_test):  
    # ********** Begin ********* #  
    logistic = LogisticRegressionCV()  
    # param说明  
    # {参数名1：[参数值1， 参数值2]，...}  
    param = {  
        'solver': ["newton-cg", "lbfgs", "sag", "saga"]  
    }  
    # 参数说明：  
    # logistic 逻辑回归分类器  
    # param_dict 超参字典  
    model = GridSearchCV(logistic, param)  
    # 使用网格搜索进行模型训练找出最佳参数  
    model.fit(x_train, y_train)  
    score = model.score(x_test, y_test)  
    # ********** End ********** # 
    return score
if __name__ == '__main__':
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
    score = model_train(x_train, x_test, y_train, y_test)
    if score >= 0.8:
        print("测试通过")
    else:
        print("测试失败")




#---------通过测试的代码在此---------#
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.datasets import load_iris


def model_train(x_train, x_test, y_train, y_test):
    # ********** Begin ********** #
    logistic = LogisticRegressionCV()
    # 设置param进行网格搜索参数设置
    # {参数名1：[参数值1， 参数值2]，...}
    param = {
        'penalty':["l1", "l2"],
    } 
    # 参数说明：
    # logistic 逻辑回归分类器
    # param_dict 超参字典
    model = GridSearchCV(logistic, param)
    # 使用网格搜索进行模型训练找出最佳参数
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    # ********* End ********** #


    return score

if __name__ == '__main__':
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

    score = model_train(x_train, x_test, y_train, y_test)

    if score >= 0.8:
        print("测试通过")
    else:
        print("测试失败")


