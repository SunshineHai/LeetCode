
'''
    使用 3个神经元，学习率是0.3，训练集是60000条数据，测试集是10000条数据，测试手写图片的识别率高达 95%。
    该神经网络一共3层：输入层、隐藏层、输出层
        输入层：784个神经元
        隐藏层：100个神经元
        输出层：10个神经元
'''

import scipy.special as sp # 里面有激活函数
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split



# 神经网络类的定义
class NeuralNetwork:

    # 初始化
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        """:arg
            inodes : 输入层 神经元个数
            hnodes : 隐藏层 神经元个数
            onodes : 输出层 神经元个数
            lr     : 学习率
            该神经网络 初始化时：
                输入层 28*28 = 784 个神经元
                隐藏层 100 个神经元
                输出层 10 个神经元
        """
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # learning rate
        self.lr = learningrate

        # 创建2个链接权重的矩阵
        '''
            self.wih ： w 代表权重，ih 分别表示 input、hidden，表示 输入层和隐藏层之间的权重
            self.who ： 表示隐藏层和输出层之间的权重
        '''
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        self.lr = learningrate

        # activate function
        self.activation_function = lambda x: sp.expit(x)
        pass

    # train the neural network:1.计算输出。2.反向传播误差
    def train(self, inputs_list:np.ndarray, targets_list:np.ndarray):
        ''':arg
            已知：样本和变量为：(n_sample, n_variable)
            inputs_list  : n_variable 输入神经元个数  输入时 维度 (784, )
            targets_list : 输出神经元个数; 本例中是 10 个。

        '''
        # 转换 inputs_list 为 二维数组
        inputs = np.array(inputs_list, ndmin=2).T        # (1, 784) --> (784, 1)
        targets = np.array(targets_list, ndmin=2).T      # 真实标签值 (10, 1)

        # 中间-隐藏层权重 * 输入神经元的值， 再通过激活函数的到隐藏层的输出值
        hidden_inputs = np.dot(self.wih, inputs)        # (100, 784) (784, 1)
        hidden_outputs = self.activation_function(hidden_inputs)    # (100, 1)

        # 隐藏层-输出层权重 * 隐藏神经元的值， 得到的值再通过激活函数
        final_inputs = np.dot(self.who, hidden_outputs)             # (10, 100) (100, 1)
        final_outputs = self.activation_function(final_inputs)      # (10, 1)
        # 反向更新权重 ： 反向传播，使用梯度下降法
        output_errors = targets - final_outputs                     # 预测值 - 实际值 得到 误差 (10, 1)
        hidden_errors = np.dot(self.who.T, output_errors)           # (100, 10) (10, 1)
        # 更新 隐藏层-输出层 权重 (10, 1) (1, 100)
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        # 更新 输入层-隐藏层权重 (100, 1) (1, 784)
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
        pass

    # 查询神经网络
    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T     # (784, 1)
        hidden_inputs = np.dot(self.wih, inputs)      # (100, 784) , (784, 1)
        hidden_outputs = self.activation_function(hidden_inputs)     # (100, 1)
        final_inputs = np.dot(self.who, hidden_outputs)              # (10, 100)
        final_outputs = self.activation_function(final_inputs)       # (10, 1)

        return final_outputs                                         # (10, 1)

    @staticmethod   # 定义普通方法，用来计算 误差
    def error(target, output):
        ''':arg
            target : 目标值/真实值
            output : 实际值/预测值
        '''

        if target.shape == (2, ):
            target = np.array(targets, ndmin=2).T      # target : (10, 1)
        if target.shape != output.shape:
            print('error: The shape bentween target and output is not equal.')
            raise ValueError
        return np.sum((target - output)**2)

    @staticmethod
    def normalization(X:np.ndarray):
        if not isinstance(X, np.ndarray):
             X = np.array(X)
        minmax = MinMaxScaler()  # 对象实例化
        X = minmax.fit_transform(X)
        return X


# 画图
def show_error(index, loss:np.ndarray):
    ''':arg
        loss : 误差损失值，(n_sample_loss, 1)
        index : 样本个数， n_sample_loss
    '''
    print(index)
    print(loss)
    try:
        if not isinstance(index, list):
            print('index 不是 list 对象')
            raise ValueError
        if not isinstance(loss, list):
            print('loss不是list对象')
            raise ValueError

        if len(index) == len(loss):
            plt.plot(index, loss)
            plt.xlabel('n_sample_index')
            plt.ylabel('loss')
            plt.legend(['loss curve'], loc='upper right')  # 显示每条线段的解释
            plt.savefig('..\\figure\\二分类.png', dpi=300)
        else:
            print("维度不同")
    except:
        print(f"index 和 loss 必须有相同的维度！")
        raise ValueError
    pass

# 输入层、隐藏层、输出层的 神经元个数
input_nodes = 7       # 28*28 个像素点，故输入层设置 784个 神经元
hidden_node = 100       # 隐藏层 先设置 100个神经元，这个数字不是最优的
output_nodes = 2       # 输出层 ：毋庸置疑 10个神经元

# 学习率，梯度下降时，学习率越小越好
learning_rate = 0.3

# 创建 神经网络对象 实例
n = NeuralNetwork(input_nodes, hidden_node, output_nodes, learning_rate)

path = r'..\data\distance.csv'
training_data_list = pd.read_csv(path).to_numpy(dtype=np.float64)   # DataFrame 对象 --> ndarray对象
#  提取 训练集、测试集
X = training_data_list[:, 1:-1]
y = training_data_list[:, -1]

# 训练集 归一化 到 (0 ~ 1)
X = n.normalization(X)

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=22)  # 训练集：测试集 = 7:3
print('x_train.length:', len(x_train))
# 训练神经网络
''':arg
    training_data_list : ['label,28*28', 'label, 28*28', ..., 'label, 28*28']
    列表中的每一个 字符串 里的数字行字符都是以 '逗号' 隔开的，因此我们根据 ',' 分割
'''
# batch_size = 2          # 每次取的样本数

Loss = []      # 存放所有误差
n_sample = []  # 存放所有样本个数


for index, record in enumerate(x_train):       # 遍历列表中的没有给字符串
    # (100, 7) 二维数据集， 开头为 标签
    inputs = record
    # 输出值 ：0.01 ~ 0.99 (0.01:表示概率最低， 0.99 表示概率最高) 以0和1表示最低和最高时，权重计算可能出现问题
    targets = np.zeros(output_nodes) + 0.01
    targets[int(y_train[index])] = 0.99           # 标签值刚好为 输出值 targets 的下标
    n.train(inputs, targets)                 # inputs 输入时维度： (784, )

    # 训练一次之后的损失：
    cnt, size = index, len(training_data_list)
    output = n.query(inputs)                # 训练一次之后的 预测值
    if index%2 == 0:
        loss, current = n.error(targets, output), index
        Loss.append(loss)
        n_sample.append(index)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        # break
        pass
    pass
show_error(n_sample, Loss)



# 测试

x_test = n.normalization(x_test)
# 测试神经网络

scorecard = []      # 计分卡


# 遍历数据集，进行测试
for index, record in enumerate(x_test):
    # 标签
    correct_label = int(y_test[index])
    # 缩放 输入结果的 区间为 (0.01 ~ 1)
    outputs = n.query(record)
    # 多维数组中的最大值
    label = np.argmax(outputs)       # 返回 (10, 1) 数组最大值的索引
    # 把 预测正确的值 添加到列表中
    if label == correct_label:
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    # break
    pass


print(len(scorecard))

# 计算准确率
scorecard_array = np.asarray(scorecard) # 把列表 转化成 numpy.ndarray 数组
# print(scorecard_array)
print('performance = ', scorecard_array.sum() / scorecard_array.size)

# performance =  0.944
