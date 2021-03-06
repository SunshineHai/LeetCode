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


pd.DataFrame(x_test).to_csv(r'C:\Users\JackYang\Desktop\x_test.csv')
pd.DataFrame(y_test).to_csv(r'C:\Users\JackYang\Desktop\y_test.csv')
pd.DataFrame(y_predict).to_csv(r'C:\Users\JackYang\Desktop\y_predict.csv')
print("--------------------score-----------------------")
print(estimator.score(x_test, y_test))              # 精度 ： 分类结果正确的样本占样本总数的比例

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

def show_error(x_name:list, precision:list, recall:list, f1:list, AUC:list, path):
    plt.plot(x_name, precision, 'or-', recall, '^g-', f1, '*b-', AUC, '.y-.')
    plt.legend(['precision', 'recall', 'f1', 'AUC'], loc='upper right')
    plt.savefig(path, dpi=300)
    plt.show()
    pass

path = r'..\figure\logicRegression.png'
x_name = ['逻辑回归']
show_error(x_name, precision, recall, f1, AUC, path)
save_path = "..\\\\figure\\\\res.png"

print("data.columns:", data.columns)

x_label = [i for i in range(len(x_train)+len(x_test))]
x_train_label, x_test_label = x_label[:len(x_train)] ,x_label[len(x_train):]
y_test_pred = y_predict

# 形参含义：(训练集样本编号, 训练集, 测试集样本编号, 测试集预测值, 全部样本编号, 全部标签值, 图片保存路径)
def draw(x_train_label, y_train, x_test_label, y_test_pred, x_label, y, save_path):

    # 画2条（0-9）的坐标轴并设置轴标签x,y
    plt.xlabel('Sample Number')
    plt.ylabel('Classification')

    colors1 = '#C0504D'
    colors2 = '#00EEEE'
    colors3 = '#000000'         # #FF6600

    area1 = np.pi * 2 ** 2
    area2 = np.pi * 3 ** 2
    area3 = np.pi * 4 ** 2

    #     print('x_test_label:', x_test_label)
    #     print('test_y:', test_y)
    #   画散点图
    plt.scatter(x_train_label, y_train, marker='^', s=area2, c=colors1, alpha=1, label='Training Set')
    # plt.scatter(x_test_label, y_test, marker='*', s=area3, c=colors2, alpha=1, label='Test Set')
    plt.scatter(x_label, y, marker='.', s=area1, c=colors3, alpha=1, label='Original data')
    plt.legend()
    plt.savefig(save_path, dpi=300)  # 指定图片像素-->高清
    plt.show()

y = pd.concat([y_train, y_test], axis=0)

draw(x_train_label, y_train, x_test_label, y_test_pred, x_label, y, save_path)
