# 自定义数据集
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd


# 1.自定义数据集
class TorchDataset(Dataset):
    ''':arg TorchDataset 继承 torch.utils.data.Dataset 类'''
    def __init__(self, path):
        # 一个由二维向量组成的数据集
        df = pd.read_csv(path)
        self.data = torch.tensor(df.iloc[:, 1: -1].to_numpy())
        self.label = torch.tensor(df.iloc[:, -1].to_numpy(dtype=np.float32))

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

    pass


# 2. 构建 Dataset 类的 实例对象
data = TorchDataset(r'data\distance.csv')

# 3.加载自定义数据集
example_loader = DataLoader(
    dataset=data,
    batch_size=2,  # 批次大小
    shuffle=False,  # False ： 数据不被随机打乱
    num_workers=0,  # 数据载入器使用的进程数目，默认为0
    drop_last=True  # 是否要把最后的批次丢弃
)
print("-----------------------------------------")
for i, (X, y) in enumerate(example_loader):
    print("i:", i)
    Data, Label = X, y
    print("data:", Data)
    print("Label:", Label)
    break