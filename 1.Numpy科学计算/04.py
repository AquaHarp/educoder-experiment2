import numpy as np


ndarray = np.ndarray


def task1(A: ndarray) -> ndarray:
    '''
    :param A n*m维的numpy数组
    任务要求：计算矩阵A的平方根并与标量2相加；
    '''
    ########## Begin ##########
    B=np.sqrt(A)+2

    ########## End ##########

    return B


def task2(A: ndarray) -> ndarray:
    '''
    :param A n*m维的numpy数组
    任务要求：将矩阵A开根号后的小数部分与原矩阵A相加；
    '''
    ########## Begin ##########
    B=np.sqrt(A)-np.floor(np.sqrt(A))+A

    ########## End ##########
    
    return B


def task3(A: ndarray) -> ndarray:
    '''
    :param A n*m维的numpy数组
    任务要求：使用通用函数numpy.dot()计算矩阵A与矩阵A转置的矢量积；
    提示：使用.T属性获得转置矩阵，例如A的转置矩阵为A.T
    '''
    ########## Begin ##########
    B=np.dot(A,A.T)

    ########## End ##########

    return B
