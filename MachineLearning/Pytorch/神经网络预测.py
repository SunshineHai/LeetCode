# 自定义数据集
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch import nn



df = pd.read_csv(r'data\distance.csv')
print(df.head())
print(df.describe())


# 1.自定义数据集
class TorchDataset(Dataset):
    ''':arg TorchDataset 继承 torch.utils.data.Dataset 类'''
    def __init__(self, path):
        # 一个由二维向量组成的数据集
        df = pd.read_csv(path)
        # 数据的类型不自定义成 float32，后续计算误差时会报错
        self.data = torch.tensor(self.normalization(df.iloc[:, 1: -1].to_numpy(dtype=np.float32)))
        self.label = torch.tensor(df.iloc[:, -1].to_numpy(dtype=np.float32))

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def normalization(X: list):
        minmax = MinMaxScaler()  # 对象实例化
        X = minmax.fit_transform(X)
        return X
    pass


# 2. 构建 Dataset 类的 实例对象
data = TorchDataset(r'data\distance.csv')

# 3.加载自定义数据集
# torch_loader = DataLoader(
#     dataset=data,
#     batch_size=2,  # 批次大小
#     shuffle=False,  # False ： 数据不被随机打乱
#     num_workers=0,  # 数据载入器使用的进程数目，默认为0
#     drop_last=True  # 是否要把最后的批次丢弃
# )
# print("-----------------------------------------")
# for batch, (X, y) in enumerate(torch_loader):
#     print("batch:", batch)
#     print(f'X.shape:{X.shape}, y.shape:{y.shape}')
#     Data, Label = X, y
#     print("data:", Data)
#     print("Label:", Label)
#     if batch == 1:
#         break

print("--------------------------------")
# 4.划分训练集和数据集  :  以 8:2 进行划分
# 乘的结果可能为小数，加 int() 强制转换为 整数， 否则小数传进去会报错
train_size, test_size = int(len(data) * 0.8), len(data) - int(len(data) * 0.8)
train_dataset, test_dataset = random_split(data, [train_size, test_size])     # train_size:518  test_size:130

batch_size = 14
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

for batch, (X, y) in enumerate(train_loader):
    print("batch:", batch)
    print(f'X.shape:{X.shape}, y.shape:{y.shape}')
    Data, Label = X, y
    print("data:", Data)
    print("Label:", Label)
    break

# 5.定义模型
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(7, 512),          # y = w*x +b
            nn.ReLU(),                  # y = Relu(y)
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2),           # 因为是二分类最后输出 2 个神经元
            # nn.Softmax(dim=1)            # 最后通过计算概率把结果变成 0/1
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    pass


# 6.机器是否可使用 GPU 训练
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# 7.训练方法
def train(dataloader, model, loss_fn, optimizer):
    ''':arg
        dataloader : dataset 实例对象的数据集
        model      : 类 NeuralNetwork 的实例对象
        loss_fn    : 损失函数，一般为计算 分类或者回归 的损失，
        optimizer  : 优化器，使用梯度下降法
    '''
    size = len(dataloader.dataset)      # dataloader 传入的数据集的样本个数
    model.train()                       # 调用 基类的 train() 方法
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)        # <class 'torch.Tensor'>
        # print("---------------train-------------------------------")
        # print(X.dtype, y.dtype)
        # print(X)
        # print(y)
        # Compute prediction error
        pred = model(X)
        # print(pred)
        # print(y)
        loss = loss_fn(pred, y.long())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# 8.测试方法
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y.long()).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# 9.实例化 神经网络模型
model = NeuralNetwork().to(device)  # 调用父类函数的 to() 方法
print(model)

# int(len(n_sample)*0.8) / batch_size : 训练集样本个数中 batch_size 有多少个?
print(len(train_loader))
# 训练集中的样本个数
print(len(train_dataset))
# 等价于上一句话
print(len(train_loader.dataset))

# 10.计算 损失值 和 优化器(使用梯度下降)
loss_fn = nn.CrossEntropyLoss()                             # 使用 交叉熵损失 计算损失
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)    # 梯度下降法; 优化器: SGD

# 11.开始训练
epochs = 5                 # 迭代次数
for t in range(epochs):
    print(f"Epoch {t + 1}\n"
          f"-------------------------------")
    train(train_loader, model, loss_fn, optimizer)
    test(test_loader, model, loss_fn)
print("Done!")