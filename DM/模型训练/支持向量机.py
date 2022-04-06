import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score  # AUC : ROC 曲线下的面积
import time


# 正常显示中文
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']

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

def show_error(x_name:list, precision:list, recall:list, f1:list, AUC:list, path):
    plt.plot(x_name, precision, 'or-', recall, '^g-', f1, '*b-', AUC, '.y-.')
    plt.legend(['precision', 'recall', 'f1', 'AUC'], loc='upper right')
    plt.savefig(path, dpi=300)
    plt.show()
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
''':arg
    criterion : 查询默认使用的邻居数
    max_depth : 决策树最大深度
'''
models = [SVC()]  # alpha 为可选项，默认 1.0，添加拉普拉修/Lidstone 平滑参数
models_str = ['SVC']
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

    # 精度：
    print("-----------------精度---------------------")
    print(model.score(x_test, y_test))  # 精度 ： 分类结果正确的样本占样本总数的比例

    pass
path = r'..\figure\show_error.tif'
x_name = models_str
print(precision_data, '\n', recall_data, '\n', f1_data, '\n', AUC_data, '\n', run_time_data)
show_error(x_name, precision_data, recall_data, f1_data, AUC_data, path)


