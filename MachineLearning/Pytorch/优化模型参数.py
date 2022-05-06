import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

# 1.加载手写识别体数据集
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()

# 超参数
# 超参数是可调整的参数，可让您控制模型优化过程。不同的超参数值会影响模型训练和收敛速度
''':arg
    Number of Epochs : 迭代数据集的次数
    Batch Size : 参数更新前通过网络传播的数据样本数
    学习率 :  在每个批次/时期更新模型参数的程度。较小的值会产生较慢的学习速度，而较大的值可能会导致训练期间出现不可预测的行为。
'''
learning_rate = 1e-3    # 学习率
batch_size = 64         # 数据样本数
epochs = 5              # 迭代次数

# 优化循环

''':arg
    一旦我们设置了超参数，我们就可以使用优化循环来训练和优化我们的模型。优化循环的每次迭代称为epoch。
    
'''

# 损失函数
''':arg
    损失函数衡量得到的结果与目标值的相异程度，是我们在训练时要最小化的损失函数。
    为了计算损失，我们使用给定数据样本的输入进行预测，并将其与真实数据标签值进行比较。
    常见的损失函数包括
    用于回归任务的 nn.MSELoss（均方误差）和 用于分类的 nn.NLLLoss（负对数似然）。
'''
# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()

# 优化器

def train_loop(dataloader, model, loss_fn, optimizer):
    ''':arg
        dataloader : 数据集
        model :      定义的模型
        loss_fn :    损失函数
        optimizer :  优化器(梯度下降)
    '''
    size = len(dataloader.dataset)      # 数据集样本个数
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)             # X : 数据集样本   y : 每个样本的标签 pred : 模型训练得到的预测值
        loss = loss_fn(pred, y)     # 传入预测值、样本标签，计算损失值，PyTorch 存储每个参数的损失梯度。

        # Backpropagation 反向传播
        optimizer.zero_grad()       # 重置模型参数的梯度，每次迭代时明确将它们归零
        loss.backward()             # 调用反向传播预测损失，PyTorch 存储每个参数的损失梯度。
        optimizer.step()            # 通过在反向传递中收集的梯度来调整参数。

        if batch % 100 == 0:        # 每 100 个样本输出一次 损失值
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)          # 计算数据集样本个数
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# for batch, (X, y) in enumerate(train_dataloader):
#     print(f"batch:{batch}")
#     print(X)
#     print(y)
# 64

loss_fn = nn.CrossEntropyLoss()                                     # 损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)   # 初始化 优化器：SGD

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n"
          f"-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
