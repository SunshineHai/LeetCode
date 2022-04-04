import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score  # AUC : ROC 曲线下的面积

# 1.获取数据
names = ['index', 'A0', 'A1', 'A2', 'A3', 'x', 'y', 'z', 'label']

data = pd.read_csv("../data/distance.csv")
# 2.基本数据处理
# 2.1 缺失值处理
data = data.replace(to_replace="?", value=np.NaN)
data = data.dropna()  # 删除至少缺少一个元素的行
print(data.head())

# 2.2 确定特征值,目标值
x = data.iloc[:, 1:8]
y = data["label"]
# 2.3 分割数据
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22)

# 3.特征工程(标准化)
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)
print('-------------------------')

# 4.机器学习(逻辑回归)
estimator = LogisticRegression()
estimator.fit(x_train, y_train)

# 5.模型评估
y_predict = estimator.predict(x_test)
print(estimator.score(x_test, y_test))

print('-----------------------------------------------')
precision = precision_score(np.array(y_test), np.array(y_predict))
recall = recall_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict)


# 0.5~1之间，越接近于1约好
# y_test = np.where(y_test > 2.5, 1, 0)   # y_test 是否 大于2.5。 为真则为1， 否则为0
AUC = roc_auc_score(y_test, y_predict)
print("AUC指标：", AUC)

""":arg
    row_name :  模型名称
"""
# 设置字体，正常显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print('-----------LogicRegression-----------------')
print(precision)
def show_error(row_name:list, precision:list, recall:list, f1:list, AUC:list, path):
#     plt.figure(figsize=(10, 10))
    plt.plot(row_name, precision, 'or-', recall, '^g-', f1, '*b-', AUC, 'bo')
    plt.legend(['precision', 'recall', 'f1', 'AUC'], loc='upper right')
    plt.savefig(path, dpi=300)
    plt.show()
pass

path = r'..\figure\evaluate_indicator.png'
show_error(['逻辑回归'], precision, recall, f1, AUC, path)

