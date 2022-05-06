## 1.构建自定义数据集

- 读取 二维表格 数据

```python
df = pd.read_csv(r'data\distance.csv')  
print(df.head())
```

out：
```text
   num           A0           A1           A2           A3   x    y   z  label
0    0  1016.931217  4782.857143  4552.962963  6298.994709  50   50  88      1
1    1  1338.424658  4920.136986  4109.178082  5943.561644  50  100  88      1
2    2  1783.790850  5033.464052  3605.882353  5619.150327  50  150  88      1
3    3  2253.617021  5218.563830  3085.159574  5322.819149  50  200  88      1
4    4  2727.142857  5370.285714  2600.142857  5033.238095  50  250  88      1
```


```python
print(df.describe())
```
out:

```text
              num           A0           A1  ...           y           z       label
count  648.000000   648.000000   648.000000  ...  648.000000  648.000000  648.000000
mean   323.500000  3822.361982  3831.485685  ...  250.000000  147.000000    0.500000
std    187.205769  1303.943989  1302.120626  ...  129.199174   42.187042    0.500386
min      0.000000   637.966102   578.874172  ...   50.000000   88.000000    0.000000
25%    161.750000  2890.911486  2868.041425  ...  150.000000  119.500000    0.000000
50%    323.500000  3948.425892  3974.006815  ...  250.000000  150.000000    0.500000
75%    485.250000  4732.334546  4783.172269  ...  350.000000  177.500000    1.000000
max    647.000000  6707.976654  6657.439614  ...  450.000000  200.000000    1.000000
```

注：mean均值 std标准差 50%中位数 25%四分位数

* 自定义数据集

```python
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
```

- 构建 Dataset 实例对象

```python
data = TorchDataset(r'data\distance.csv')
```


- 加载自定义数据集

```python
torch_loader = DataLoader(  
    dataset=data,  
    batch_size=2,  # 批次大小  
    shuffle=False,  # False ： 数据不被随机打乱  
    num_workers=0,  # 数据载入器使用的进程数目，默认为0  
    drop_last=True  # 是否要把最后的批次丢弃  
)
```


- 输出自定义数据集

```python
for batch, (X, y) in enumerate(torch_loader):  
    print("batch:", batch)  
    print(f'X.shape:{X.shape}, y.shape:{y.shape}')  
    Data, Label = X, y  
    print("data:", Data)  
    print("Label:", Label)  
    if batch == 1:  
        break
```

out:
```text
batch: 0
X.shape:torch.Size([14, 7]), y.shape:torch.Size([14])
data: tensor([[0.2451, 0.6446, 0.4407, 0.7390, 0.1250, 0.3750, 0.0000],
        [0.9156, 0.5625, 0.6628, 0.1021, 1.0000, 0.8750, 1.0000],
        [0.6445, 0.4457, 0.5700, 0.3105, 0.7500, 0.6250, 0.3750],
        [0.8122, 0.3893, 0.7183, 0.2404, 1.0000, 0.6250, 0.7321],
        [0.6144, 0.5965, 0.3801, 0.3971, 0.5000, 0.7500, 1.0000],
        [0.4900, 0.7703, 0.1859, 0.6201, 0.1250, 0.7500, 0.7321],
        [0.7527, 0.7540, 0.3163, 0.3293, 0.5000, 1.0000, 0.7321],
        [0.5562, 0.1616, 0.8498, 0.6476, 0.8750, 0.1250, 0.3750],
        [0.0704, 0.6935, 0.6534, 1.0000, 0.0000, 0.0000, 1.0000],
        [0.2204, 0.3961, 0.7277, 0.8184, 0.3750, 0.0000, 0.3750],
        [0.0669, 0.6940, 0.6085, 0.9276, 0.0000, 0.1250, 0.7321],
        [0.5954, 0.7183, 0.3396, 0.4607, 0.3750, 0.7500, 0.0000],
        [0.6543, 0.2643, 0.7265, 0.4357, 0.8750, 0.3750, 1.0000],
        [0.7997, 0.7694, 0.4346, 0.2439, 0.6250, 1.0000, 0.7321]])
Label: tensor([0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 1., 1., 0., 1.])
```

注：DataLoader 中 ```batch_size=2``` 表示批大小=2，即一次从数据集中，取出2个样本，batch_size 可根据情况设定。


## 2.划分数据集

- 划分训练集和数据集  :  以 8:2 进行划分：

```python
# 乘的结果可能为小数，加 int() 强制转换为 整数， 否则小数传进去会报错  
train_size, test_size = int(len(data) * 0.8), len(data) - int(len(data) * 0.8)  
train_dataset, test_dataset = random_split(data, [train_size, test_size])     # train_size:518  test_size:130
```

- 加载数据集

```python
batch_size = 14  
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)  
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
```

- 测试输出数据集

```python
for batch, (X, y) in enumerate(train_loader):  
    print("batch:", batch)  
    print(f'X.shape:{X.shape}, y.shape:{y.shape}')  
    Data, Label = X, y  
    print("data:", Data)  
    print("Label:", Label)  
    break
```

out:
```python
batch: 0
X.shape:torch.Size([14, 7]), y.shape:torch.Size([14])
data: tensor([[0.2451, 0.6446, 0.4407, 0.7390, 0.1250, 0.3750, 0.0000],
        [0.9156, 0.5625, 0.6628, 0.1021, 1.0000, 0.8750, 1.0000],
        [0.6445, 0.4457, 0.5700, 0.3105, 0.7500, 0.6250, 0.3750],
        [0.8122, 0.3893, 0.7183, 0.2404, 1.0000, 0.6250, 0.7321],
        [0.6144, 0.5965, 0.3801, 0.3971, 0.5000, 0.7500, 1.0000],
        [0.4900, 0.7703, 0.1859, 0.6201, 0.1250, 0.7500, 0.7321],
        [0.7527, 0.7540, 0.3163, 0.3293, 0.5000, 1.0000, 0.7321],
        [0.5562, 0.1616, 0.8498, 0.6476, 0.8750, 0.1250, 0.3750],
        [0.0704, 0.6935, 0.6534, 1.0000, 0.0000, 0.0000, 1.0000],
        [0.2204, 0.3961, 0.7277, 0.8184, 0.3750, 0.0000, 0.3750],
        [0.0669, 0.6940, 0.6085, 0.9276, 0.0000, 0.1250, 0.7321],
        [0.5954, 0.7183, 0.3396, 0.4607, 0.3750, 0.7500, 0.0000],
        [0.6543, 0.2643, 0.7265, 0.4357, 0.8750, 0.3750, 1.0000],
        [0.7997, 0.7694, 0.4346, 0.2439, 0.6250, 1.0000, 0.7321]])
Label: tensor([0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 1., 1., 0., 1.])
```

## 3.定义模型

```python
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
            # nn.Softmax(dim=1)            # 最后通过计算概率把结果变成 0/1        )  
  
    def forward(self, x):  
        x = self.flatten(x)  
        logits = self.linear_relu_stack(x)  
        return logits  
    pass
```


## 4.机器是否可使用 GPU 训练

```python
device = "cuda" if torch.cuda.is_available() else "cpu"  
print(f"Using {device} device")
```

out:
```text
Using cuda device
```

## 5.训练方法

```python
def train(dataloader, model, loss_fn, optimizer):  
    ''':arg  
        dataloader : dataset 实例对象的数据集  
        model      : 类 NeuralNetwork 的实例对象  
        loss_fn    : 损失函数，一般为计算 分类或者回归 的损失，  
        optimizer  : 优化器，使用梯度下降法  
    '''    size = len(dataloader.dataset)      # dataloader 传入的数据集的样本个数  
    model.train()                       # 调用 基类的 train() 方法  
    for batch, (X, y) in enumerate(dataloader):  
        X, y = X.to(device), y.to(device)        # <class 'torch.Tensor'>  
        # print("---------------train-------------------------------")        # print(X.dtype, y.dtype)        # print(X)        # print(y)        # Compute prediction error        pred = model(X)  
        # print(pred)  
        # print(y)        loss = loss_fn(pred, y.long())  
  
        # Backpropagation  
        optimizer.zero_grad()  
        loss.backward()  
        optimizer.step()  
  
        if batch % 10 == 0:  
            loss, current = loss.item(), batch * len(X)  
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
```

## 6.测试方法

```python
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
```

## 7.实例化模型

```python
model = NeuralNetwork().to(device)  # 调用父类函数的 to() 方法  
print(model)
```
out:
```text
NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=7, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=2, bias=True)
  )
)
```


## 8.损失与优化器

```python
loss_fn = nn.CrossEntropyLoss()                             # 使用 交叉熵损失 计算损失  
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)    # 梯度下降法; 优化器: SGD
```

## 9. 开始训练

```python
epochs = 100                 # 迭代次数  
for t in range(epochs):  
    print(f"Epoch {t + 1}\n"  
          f"-------------------------------")  
    train(train_loader, model, loss_fn, optimizer)  
    test(test_loader, model, loss_fn)  
print("Done!")
```

out:
```text
Epoch 1
-------------------------------
loss: 0.691605  [    0/  518]
loss: 0.687310  [  140/  518]
loss: 0.695601  [  280/  518]
loss: 0.695044  [  420/  518]
Test Error: 
 Accuracy: 46.9%, Avg loss: 0.693291 

Epoch 2
-------------------------------
loss: 0.691656  [    0/  518]
loss: 0.687376  [  140/  518]
loss: 0.695556  [  280/  518]
loss: 0.694968  [  420/  518]
Test Error: 
 Accuracy: 46.9%, Avg loss: 0.693299 

Epoch 3
-------------------------------
loss: 0.691695  [    0/  518]
loss: 0.687428  [  140/  518]
loss: 0.695517  [  280/  518]
loss: 0.694901  [  420/  518]
Test Error: 
 Accuracy: 47.7%, Avg loss: 0.693306 

Epoch 4
-------------------------------
loss: 0.691731  [    0/  518]
loss: 0.687467  [  140/  518]
loss: 0.695482  [  280/  518]
loss: 0.694830  [  420/  518]
Test Error: 
 Accuracy: 47.7%, Avg loss: 0.693311 

Epoch 5
-------------------------------
loss: 0.691761  [    0/  518]
loss: 0.687483  [  140/  518]
loss: 0.695449  [  280/  518]
loss: 0.694756  [  420/  518]
Test Error: 
 Accuracy: 47.7%, Avg loss: 0.693317 

Done!
```