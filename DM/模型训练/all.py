import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb                                  # LightGBM
from sklearn.svm import SVC                             # 支持向量机回归
from sklearn.neighbors import KNeighborsClassifier      # K近邻算法
from sklearn.naive_bayes import MultinomialNB           # 朴素贝叶斯
from sklearn.neural_network import MLPClassifier        # 神经网络分类
from sklearn.tree import DecisionTreeClassifier         # 决策树分类
from sklearn.linear_model import LogisticRegression     # 逻辑回归


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score  # AUC : ROC 曲线下的面积



import warnings

warnings.filterwarnings('ignore')
# 正常显示中文
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
# 正常显示符号
# matplotlib 画图中中文显示会有问题，需要这两行设置默认字体
from matplotlib import rcParams
rcParams['axes.unicode_minus'] = False

from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import csv
import re
from openpyxl import load_workbook  # 读取 xlsx 类型的文件需要专门的读取程序


# 数据标准化
def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma



def normalization(X):
    """X : ndarray 对象"""
    min_ = X.min(axis=0)        # 得到的是 Series 对象
    max_ = X.max(axis=0)
    row_num = X.shape[0]        # row_num 是X的行数
    ranges = max_ - min_
    print(ranges)
    molecule = X - np.tile(min_, (row_num, 1))     # np.tile(A, (row, col) ) :把A(看做整体)复制row行，col 列,分子:X-min
    denominator = np.tile(ranges, (row_num, 1))    # 分母 max-min
    X = molecule/denominator
    return X, min_, ranges


# 形参含义：(训练集样本编号, 训练集, 测试集样本编号, 测试集预测值, 全部样本编号, 全部标签值, 图片保存路径)
def draw(x_train_label, y_train, x_test_label, y_test_pred, x_label, y, save_path):

    # 画2条（0-9）的坐标轴并设置轴标签x,y
    plt.xlabel('Sample Number')
    plt.ylabel('Classification')

    colors1 = '#C0504D'
    colors2 = '#00EEEE'
    colors3 = '#FF6600'

    area1 = np.pi * 2 ** 2
    area2 = np.pi * 3 ** 2
    area3 = np.pi * 4 ** 2

    #     print('x_test_label:', x_test_label)
    #     print('test_y:', test_y)
    #   画散点图
    plt.scatter(x_train_label, y_train, marker='^', s=area2, c=colors1, alpha=1, label='Training Set')
    plt.scatter(x_test_label, y_test, marker='*', s=area3, c=colors2, alpha=1, label='Test Set')
    plt.scatter(x_label, y, marker='o', s=area1, c=colors3, alpha=1, label='Original data')
    plt.legend()
    plt.savefig(save_path, dpi=300)  # 指定图片像素-->高清
    plt.show()

""":arg
    \d : 等价于 [0-9]
    \. : 匹配 . 
"""
def p_words(string):
    string_list = re.findall(r"\d+\.\d+", string)  # 找到 string 里与 pattern 匹配的字串
    return string_list[0]


def show_error(x_name, precision, recall, f1, AUC, path):
    plt.plot(x_name, precision, 'or-', recall, '^g-', f1, '*b-', AUC, '.y-.')
    plt.legend(['precision', 'recall', 'f1', 'AUC'], loc='upper right')
    plt.savefig(path, dpi=300)
    plt.show()
    pass


pass
# 1.获取数据
path = r"..\data\distance.csv"
all_data = pd.read_csv(path)

# 2.基本数据处理

# 2.1 缺失值处理
# 这里缺失值已经进行处理过

# 2.2 确定特征值 , 目标值
# data_label = all_data[:, 0]  # 第一列是索引 [0-647] 是样本个数
x = all_data.iloc[:, 1:-1]
y = all_data.iloc[:, -1]     # 分类的标签 [0, 1]

# 2.3 分割数据
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22)

# 3. 特征工程(归一化)
data, _, _ = normalization(x)  # 进行归一化

# 4.机器学习

models = [lgb.LGBMClassifier(objective='binary', num_leaves=31, learning_rate=0.05, n_estimators=20),
          KNeighborsClassifier(n_neighbors=9), DecisionTreeClassifier(criterion="entropy", max_depth=5),
          MultinomialNB(alpha=1), SVC(), MLPClassifier(alpha=0.01), LogisticRegression()]
models_str = ['LightGBM', 'KNN', 'DTree',
              'naive_bayes', 'SVC', 'MLP', 'LR']
precision_data = []
recall_data = []
f1_data = []
AUC_data = []
run_time_data = []
for name, model in zip(models_str, models):  # zip() : 变量2个迭代器
    print('开始训练模型:' + name)
    model = model  # 建立模型
    model.fit(x_train, y_train)
    startTime = time.time()
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    stopTime = time.time()

    save_path = "..\\\\figure\\" + name + ".png"
    x_train_label = [x for x in range(len(y_train))]
    x_test_label = [x for x in range(len(y_train), len(y_train)+len(y_test_pred))]
    x_label = [x for x in range(len(x))]
    # draw(x_train_label, y_train, x_test_label, y_test_pred, x_label, y, save_path)

    print('-----------------------------------------------')
    run_time = stopTime - startTime
    precision = precision_score(np.array(y_test), np.array(y_test_pred))
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    AUC = roc_auc_score(y_test, y_test_pred)

    print('The precision is :', precision)
    print('The recall is :', recall)
    print('The f1 is:', f1)
    print("AUC指标：", AUC)
    print("run_time:", run_time)

    precision_data.append(precision)
    recall_data.append(recall)
    f1_data.append(f1)
    AUC_data.append(AUC)
    run_time_data.append(run_time)
path = r'..\figure\show_error.tif'
x_name = models_str
print(precision_data, '\n', recall_data, '\n', f1_data, '\n', AUC_data, '\n', run_time_data)
show_error(x_name, precision_data, recall_data, f1_data, AUC_data, path)

