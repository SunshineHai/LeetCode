# 多分类-mnist数据集-Pytorch实现



## model.parameters()
```python
for param in model.parameters():              # model.parameters() 返回 generate 迭代器
    print(type(param), param.size())
```

out：

```text
<class 'torch.nn.parameter.Parameter'> torch.Size([100, 784])
<class 'torch.nn.parameter.Parameter'> torch.Size([100])
<class 'torch.nn.parameter.Parameter'> torch.Size([10, 100])
<class 'torch.nn.parameter.Parameter'> torch.Size([10])
```

由 model.parameters() 的输出结果，可以看出返回的是 各个层的 权重w 和 偏置 b。

在 单个训练集循环中，模型在训练集上做出预测，并反向更新(backpropagates)预测误差来调整模型的参数。

## 下载数据集

```python
training_data = datasets.MNIST(
    root='data',        # 数据下载后存储的根目录文件夹名称
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.MNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor(),
)
```

## 加载数据集

```python
batch_size = 60  # 初始化训练轮数
# 加载 mnist 数据集 : 返回 containing batch_size=10 features and labels respectively
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)  # 共60000条数据
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)       # 共10000条数据
```

测试：

```python
for X, y in test_dataloader:
    print(f"Shape of X[N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.size()}")
    break
```

out：

```text
Shape of X[N, C, H, W]: torch.Size([60, 1, 28, 28])
Shape of y: torch.Size([60])
```

## 机器是否支持GPU

```python
# 训练集是否使用 CPU 或者 GPU device 进行训练
device = "cuda" if torch.cuda.is_available else "cpu"
```

## 定义神经网络

```python
class NeuralNetwork(nn.Module):
    """:arg
        在Pytorch中定义神经网络，我们创建 NeuralNetwork 类 继承nn.Module。我们定义在 __init__ 函数中定义网络的层数，
        在 forword() 中明确 数据怎样通过 网络。为了加速神经网络的训练，如果GPU可用就把它放到GPU上。

        init : 初始化神经网络的曾说以及训练数据要通过的方法
    """
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 30),  # z = x·w^T + b, 参数表示：(输入层神经元个数, 输出层神经元个数)
            nn.Sigmoid(),
            nn.Linear(30, 10),
            nn.Sigmoid()
        )

    # 前向传播
    def forward(self, x):
        # print('-'*6 + 'forward' + '-'*6)
        x = self.flatten(x)                 # 把 (1, 28, 28) 转为 (1, 784)
        # print('----flatten()-----')
        # print(x.shape)
        logits = self.linear_relu_stack(x)
        return logits
    pass
```



## 使用GPU进行训练

```python
model = NeuralNetwork().to(device)   # 使用 GPU 训练
print(model)
```

## 优化模型参数

```python
loss_fn = nn.CrossEntropyLoss()       # 交叉熵损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=3.0)
```

## 训练

```python
def train(dataloader, model, loss_fn, optimizer):
    """:arg
        dataloader : 含有 batch_size 个训练集
        model : 神经网络类的实例
        loss_fn : 损失函数
        optimizer: 优化器，使用随机梯度下降算法，反向传播误差更新权重和偏置
    """
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # 计算激活层误差
        pred = model(X)
        loss = loss_fn(pred, y)

        # 反向传播 (Backpropagation)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        pass
    pass
```

## 测试

```python
def test(dataloader, model, loss_fn):
    ''':arg
        We also check the model’s performance against the test dataset to ensure it is learning.
    '''
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    pass
```

## 开始训练测试

```python
# epochs ： 迭代次数
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")
```

## 保存模型

```python
# 保存模型
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
```

## 加载模型

```python
model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = pred[0].argmax(0), y
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
```

## 全部代码

```python
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

from torch.utils.data import DataLoader
import utils
import torch.nn as nn

# 下载数据集
training_data = datasets.MNIST(
    root='data',        # 数据下载后存储的根目录文件夹名称
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.MNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 60  # 初始化训练轮数
# 加载 mnist 数据集 : 返回 containing batch_size=10 features and labels respectively
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)  # 共60000条数据
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)       # 共10000条数据

# training_data[0] : ((1, 28, 28), label)
for X, y in test_dataloader:
    print(f"Shape of X[N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.size()}")
    break

# utils.draw_sample_image(training_data[0][0])

# 训练集是否使用 CPU 或者 GPU device 进行训练
device = "cuda" if torch.cuda.is_available else "cpu"

# 定义神经网络 (28*28, 10)
class NeuralNetwork(nn.Module):
    """:arg
        在Pytorch中定义神经网络，我们创建 NeuralNetwork 类 继承nn.Module。我们定义在 __init__ 函数中定义网络的层数，
        在 forword() 中明确 数据怎样通过 网络。为了加速神经网络的训练，如果GPU可用就把它放到GPU上。

        init : 初始化神经网络的曾说以及训练数据要通过的方法
    """
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 30),  # z = x·w^T + b, 参数表示：(输入层神经元个数, 输出层神经元个数)
            nn.Sigmoid(),
            nn.Linear(30, 10),
            nn.Sigmoid()
        )

    # 前向传播
    def forward(self, x):
        # print('-'*6 + 'forward' + '-'*6)
        x = self.flatten(x)                 # 把 (1, 28, 28) 转为 (1, 784)
        # print('----flatten()-----')
        # print(x.shape)
        logits = self.linear_relu_stack(x)
        return logits
    pass


model = NeuralNetwork().to(device)   # 使用 GPU 训练
print(model)


# X = torch.rand(1, 28, 28, device=device)
# print("------logits------")
# logits = model(X)   # 执行 forward() 方法
# print(logits)
# print(logits.shape)
#
# # 模型层数
#
# input_image = torch.rand(3, 28, 28)
# print(input_image.size())

# 初始化超参数(层数、每层神经元个数，训练轮数)

# 测试网络性能

# 优化模型参数
loss_fn = nn.CrossEntropyLoss()       # 交叉熵损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=3.0)
print('---------------------------')
# for param in model.parameters():
#     print(type(param), param.size())


def train(dataloader, model, loss_fn, optimizer):
    """:arg
        dataloader : 含有 batch_size 个训练集
        model : 神经网络类的实例
        loss_fn : 损失函数
        optimizer: 优化器，使用随机梯度下降算法，反向传播误差更新权重和偏置
    """
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # 计算激活层误差
        pred = model(X)
        loss = loss_fn(pred, y)

        # 反向传播 (Backpropagation)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        pass
    pass


def test(dataloader, model, loss_fn):
    ''':arg
        We also check the model’s performance against the test dataset to ensure it is learning.
    '''
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    pass


# epochs ： 迭代次数
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

# 保存模型
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

# 加载模型
model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = pred[0].argmax(0), y
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
```

out:

```
D:\Users\JackYang\anaconda3\envs\pytorch\python.exe D:/SunshineHai/NerualNetworkUsePytorch/network.py
Shape of X[N, C, H, W]: torch.Size([60, 1, 28, 28])
Shape of y: torch.Size([60])
NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=30, bias=True)
    (1): Sigmoid()
    (2): Linear(in_features=30, out_features=10, bias=True)
    (3): Sigmoid()
  )
)
---------------------------
Epoch 1
-------------------------------
loss: 2.306238  [    0/60000]
loss: 1.758612  [ 6000/60000]
loss: 1.684784  [12000/60000]
loss: 1.668170  [18000/60000]
loss: 1.585483  [24000/60000]
loss: 1.619343  [30000/60000]
loss: 1.541026  [36000/60000]
loss: 1.559132  [42000/60000]
loss: 1.584110  [48000/60000]
loss: 1.530745  [54000/60000]
Test Error: 
 Accuracy: 91.3%, Avg loss: 1.555860 

Epoch 2
-------------------------------
loss: 1.587200  [    0/60000]
loss: 1.570800  [ 6000/60000]
loss: 1.557328  [12000/60000]
loss: 1.527586  [18000/60000]
loss: 1.565426  [24000/60000]
loss: 1.520149  [30000/60000]
loss: 1.578616  [36000/60000]
loss: 1.542387  [42000/60000]
loss: 1.525916  [48000/60000]
loss: 1.563138  [54000/60000]
Test Error: 
 Accuracy: 92.5%, Avg loss: 1.536038 

Epoch 3
-------------------------------
loss: 1.507710  [    0/60000]
loss: 1.505544  [ 6000/60000]
loss: 1.523451  [12000/60000]
loss: 1.551916  [18000/60000]
loss: 1.528533  [24000/60000]
loss: 1.519681  [30000/60000]
loss: 1.523500  [36000/60000]
loss: 1.534618  [42000/60000]
loss: 1.537771  [48000/60000]
loss: 1.543024  [54000/60000]
Test Error: 
 Accuracy: 93.3%, Avg loss: 1.527381 

Epoch 4
-------------------------------
loss: 1.539762  [    0/60000]
loss: 1.484228  [ 6000/60000]
loss: 1.546722  [12000/60000]
loss: 1.551078  [18000/60000]
loss: 1.514289  [24000/60000]
loss: 1.538691  [30000/60000]
loss: 1.490507  [36000/60000]
loss: 1.567897  [42000/60000]
loss: 1.515709  [48000/60000]
loss: 1.512754  [54000/60000]
Test Error: 
 Accuracy: 93.9%, Avg loss: 1.522638 

Epoch 5
-------------------------------
loss: 1.531539  [    0/60000]
loss: 1.555603  [ 6000/60000]
loss: 1.502708  [12000/60000]
loss: 1.503383  [18000/60000]
loss: 1.516106  [24000/60000]
loss: 1.532699  [30000/60000]
loss: 1.549412  [36000/60000]
loss: 1.512203  [42000/60000]
loss: 1.505752  [48000/60000]
loss: 1.503409  [54000/60000]
Test Error: 
 Accuracy: 94.2%, Avg loss: 1.519839 

Done!
Saved PyTorch Model State to model.pth
Predicted: "7", Actual: "7"

Process finished with exit code 0

```

