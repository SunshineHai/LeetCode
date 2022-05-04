
# 构建神经网络
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 判断机器是否含有GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# 定义神经网络类
class NeuralNetwork(nn.Module):                 # 继承 父类 nn.Module
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

    def forward(self, x):               # forward 方法：实现对输入数据的操操作
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# 创建实例
model = NeuralNetwork().to(device)
print(model)


X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

# 模型层

# 生成 3张大小为 28X28 的图像的小批量样本
input_image = torch.rand(3, 28, 28)
print(input_image.size())

# nn.flatten()
# 把 3x28x28 维   -->  3x784
flatten = nn.Flatten()              # 维度转化处理
flat_image = flatten(input_image)
print(flat_image.size())            # 维度：3x784       (n_sample, n_varible) 3个样本， 784个变量
print(flat_image)


# nn.linear() : y = xA^T + b  用来进行矩阵的相乘，对输入数据进行线性变换
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

# nn.ReLU ：激活函数
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

# nn.Sequential : 是一个有序的模块容器，数据按照定义的顺序通过所有模块。
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3, 28, 28)
logits = seq_modules(input_image)
print(logits)

# nn.Softmax : 神经网络最后一层所使用的函数
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)

# 模型参数
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
