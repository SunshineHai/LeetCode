# from Predict.lstm_reg import *
import torch
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# 1.归一化方法
def normalization(X : list):
    minmax = MinMaxScaler()  # 对象实例化
    X = minmax.fit_transform(X)
    return X


# 2.excel 表格数据 转换成 Pytorch 可以识别的 数据集
def init_dataset(X:np.ndarray, y:np.ndarray, batch_size:int):
    ''':arg
        X : (n_sample, n_variable)
        y : label
        batch_size : 1批 batch_size个样本，相当于把 训练集每 batch_size个分为一批 进行训练
    '''
    # 判断 X 、 y 的类型是否为 np.ndarray
    if not isinstance(X, np.ndarray):
        X, y = X.to_numpy(), np.array(y)
        # raise TypeError

    # 判断 y 的数据类型是否为 浮点型
    if not isinstance(y.dtype, np.float32):
        y.dtype = np.float32
        # raise TypeError
    # numpy 转化成 torch 格式的 张量
    X = torch.from_numpy(X).type(torch.FloatTensor)
    y = torch.from_numpy(y).type(torch.LongTensor)
    # 1.X 归一化
    X = normalization(X)
    # print(X)
    # 2.划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(input, output, test_size=0.2)

    # 3.numpy 转换为 Tensor(2者功能一样，只是 Pytorch 中的Tensor 类型可以在 GPU 上运行)
    X_train, y_train = torch.Tensor(X_train), torch.Tensor(y_train)

    # 4.转换成Torch能识别的 DataSet
    torch_dataset = TensorDataset(X_train, y_train)

    # 5.把 dataset 放入 DataLoader
    train_dataloader = DataLoader(
        dataset=torch_dataset,              # 数据集
        batch_size=batch_size,              # 批大小
        shuffle=False    # 要不要打乱数据
    )

    # 测试集的转换
    X_test, y_test = torch.Tensor(X_test), torch.Tensor(y_test)
    torch_dataset = TensorDataset(X_test, y_test)
    test_dataloader = DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=False
     )
    return train_dataloader, test_dataloader


# 定义模型
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


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

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


if __name__ == '__main__':

    # 是否使用 GPU 训练
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    path = r'data\distance.csv'
    data = pd.read_csv(path)
    input = data.iloc[:, 1:-1]
    output = data.iloc[:, -1]


    batch_size = 14
    input, output = input.to_numpy(), output.to_numpy(dtype=np.float32)

    train_dataloader, test_dataloader = init_dataset(input, output, batch_size=batch_size)
    # n_sample : 648  batch_size:16 train:test=8:2 故 len(X_train)=648*0.8=518 ?batch_size = 518/16=32 ;
    # 648*0.2/16=8

    for X, y in train_dataloader:
        print(f"Shape of X[N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    model = NeuralNetwork().to(device)  # 调用父类函数的 to() 方法
    print(model)
    #
    # 优化模型参数
    # 为了训练一个模型，我们需要一个损失函数 和一个优化器。
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    epochs = 5  # 迭代次数
    for t in range(epochs):
        print(f"Epoch {t + 1}\n"
              f"-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")

    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")

    # 加载模型
    model = NeuralNetwork()
    model.load_state_dict(torch.load("model.pth"))
    #
    # classes = [
    #     "T-shirt/top",
    #     "Trouser",
    #     "Pullover",
    #     "Dress",
    #     "Coat",
    #     "Sandal",
    #     "Shirt",
    #     "Sneaker",
    #     "Bag",
    #     "Ankle boot",
    # ]
    #
    # # 该模型现在可用于进行预测
    # model.eval()
    # x, y = test_data[0][0], test_data[0][1]
    # with torch.no_grad():
    #     pred = model(x)
    #     predicted, actual = classes[pred[0].argmax(0)], classes[y]
    #     print(f'Predicted: "{predicted}", Actual: "{actual}"')

    # model = NeuralNetwork().to(device)
    # print(model)

    # X = torch.rand(1, 28, 28, device=device)
    # logits = model(X)
    # pred_probab = nn.Softmax(dim=1)(logits)
    # y_pred = pred_probab.argmax(1)
    # print(f"Predicted class: {y_pred}")


