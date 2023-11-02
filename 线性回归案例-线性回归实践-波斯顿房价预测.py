#encoding=utf8
#********* Begin *********#
import pandas as pd
from sklearn.linear_model import LinearRegression
#获取训练数据
train_data = pd.read_csv('./step3/train_data.csv')
#获取训练标签
train_label = pd.read_csv('./step3/train_label.csv')
train_label = train_label['target']
#获取测试数据
test_data = pd.read_csv('./step3/test_data.csv')
lr = LinearRegression()
#训练模型
lr.fit(train_data,train_label)
#获取预测标签
predict = lr.predict(test_data)
#将预测标签写入csv
df = pd.DataFrame({'result':predict}) 
df.to_csv('./step3/result.csv', index=False)
#********* End *********#
 
 

