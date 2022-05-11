## 1.加载数据集
我们使用 FashionMNIST 数据集。
注：FashionMNIST 数据集 是一个定位在比MNIST图片识别问题稍复杂的数据集,它的设定与MNIST几乎完全一样,包含了 10 类不同类型的衣服、鞋子、包等灰度图片。

以下实例是多分类问题。

- 使用 TorchVision 数据集

```python
# 从开源数据集中下载训练集: 这里使用 FashionMNIST 数据集
training_data = datasets.FashionMNIST(  
    root="data",  
    train=True,  
    download=True,  
    transform=ToTensor(),  
)

# 下载 测试集  
test_data = datasets.FashionMNIST(  
    root="data",  
    train=False,  
    download=True,  
    transform=ToTensor(),  
)
```


- 加载数据集

```python
batch_size = 64  
# create data loaders  
train_dataloader  = DataLoader(training_data, batch_size=batch_size)  
test_dataloader = DataLoader(test_data, batch_size=batch_size)  
  
for X, y in test_dataloader:  
    print(f"Shape of X[N, C, H, W]: {X.shape}")  
    print(f"Shape of y: {y.shape} {y.dtype}")  
    break
```

## 2. 创建模型

```python
# Get cpu or guv device for training  
device = "cuda" if torch.cuda.is_available() else "cpu"  
print(f"Using {device} device")  
  
  
# define model1  
class NeuralNetwork(nn.Module):  
    def __init__(self):  
        super(NeuralNetwork, self).__init__()  
        self.flatten = nn.Flatten()  
        self.linear_relu_stack = nn.Sequential(  
            nn.Linear(28*28, 512),  
            nn.ReLU(),  
            nn.Linear(512, 512),  
            nn.ReLU(),  
            nn.Linear(512, 10)  
        )  
  
    def forward(self, x):  
        x = self.flatten(x)  
        logits = self.linear_relu_stack(x)  
        return logits  
    pass  
  
  
  
model = NeuralNetwork().to(device)    # 调用父类函数的 to() 方法  
print(model)
```


## 3.优化模型参数

```python
# 为了训练一个模型，我们需要一个损失函数 和一个优化器。  
loss_fn = nn.CrossEntropyLoss()  
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
```


## 4.训练和测试模型

```python
def train(dataloader, model, loss_fn, optimizer):  
    size = len(dataloader.dataset)  
    model.train()  
    for batch, (X, y) in enumerate(dataloader):  
        X, y = X.to(device), y.to(device)  
  
        # Compute prediction error  
        pred = model(X)  
        loss = loss_fn(pred, y)  
  
        # Backpropagation  
        optimizer.zero_grad()  
        loss.backward()  
        optimizer.step()  
  
        if batch % 100 == 0:  
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
            test_loss += loss_fn(pred, y).item()  
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()  
    test_loss /= num_batches  
    correct /= size  
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```

## 5. 开始训练

```python
epochs = 5              # 迭代次数  
for t in range(epochs):  
    print(f"Epoch {t+1}\n-------------------------------")  
    train(train_dataloader, model, loss_fn, optimizer)  
    test(test_dataloader, model, loss_fn)  
print("Done!")
```

## 6.保存模型

```python
torch.save(model.state_dict(), "model.pth")  
print("Saved PyTorch Model State to model.pth")
```

## 7.加载模型

```python
model = NeuralNetwork()  
model.load_state_dict(torch.load("model.pth"))
```

## 8.预测

```python
classes = [  
    "T-shirt/top",  
    "Trouser",  
    "Pullover",  
    "Dress",  
    "Coat",  
    "Sandal",  
    "Shirt",  
    "Sneaker",  
    "Bag",  
    "Ankle boot",  
]  
  
# 该模型现在可用于进行预测  
model.eval()  
x, y = test_data[0][0], test_data[0][1]  
with torch.no_grad():  
    pred = model(x)  
    predicted, actual = classes[pred[0].argmax(0)], classes[y]  
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
```


运行结果：

```text
Connected to pydev debugger (build 201.7846.77)
Shape of X[N, C, H, W]: torch.Size([64, 1, 28, 28])
Shape of y: torch.Size([64]) torch.int64
Using cuda device
NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)
Epoch 1
-------------------------------
loss: 2.302747  [    0/60000]
loss: 2.296829  [ 6400/60000]
loss: 2.277063  [12800/60000]
loss: 2.274143  [19200/60000]
loss: 2.257148  [25600/60000]
loss: 2.220658  [32000/60000]
loss: 2.236387  [38400/60000]
loss: 2.194870  [44800/60000]
loss: 2.189612  [51200/60000]
loss: 2.161722  [57600/60000]
Test Error: 
 Accuracy: 48.0%, Avg loss: 2.157671 

Epoch 2
-------------------------------
loss: 2.163260  [    0/60000]
loss: 2.158785  [ 6400/60000]
loss: 2.100264  [12800/60000]
loss: 2.121629  [19200/60000]
loss: 2.077504  [25600/60000]
loss: 2.004310  [32000/60000]
loss: 2.042329  [38400/60000]
loss: 1.955108  [44800/60000]
loss: 1.961550  [51200/60000]
loss: 1.885384  [57600/60000]
Test Error: 
 Accuracy: 57.4%, Avg loss: 1.891497 

Epoch 3
-------------------------------
loss: 1.921233  [    0/60000]
loss: 1.894850  [ 6400/60000]
loss: 1.776564  [12800/60000]
loss: 1.820165  [19200/60000]
loss: 1.725456  [25600/60000]
loss: 1.653135  [32000/60000]
loss: 1.686808  [38400/60000]
loss: 1.579594  [44800/60000]
loss: 1.608693  [51200/60000]
loss: 1.490921  [57600/60000]
Test Error: 
 Accuracy: 60.7%, Avg loss: 1.519821 

Epoch 4
-------------------------------
loss: 1.587000  [    0/60000]
loss: 1.551294  [ 6400/60000]
loss: 1.401518  [12800/60000]
loss: 1.474207  [19200/60000]
loss: 1.372276  [25600/60000]
loss: 1.343572  [32000/60000]
loss: 1.372158  [38400/60000]
loss: 1.289430  [44800/60000]
loss: 1.328003  [51200/60000]
loss: 1.217984  [57600/60000]
Test Error: 
 Accuracy: 63.6%, Avg loss: 1.252232 

Epoch 5
-------------------------------
loss: 1.331331  [    0/60000]
loss: 1.308383  [ 6400/60000]
loss: 1.145954  [12800/60000]
loss: 1.251308  [19200/60000]
loss: 1.140415  [25600/60000]
loss: 1.146848  [32000/60000]
loss: 1.182463  [38400/60000]
loss: 1.111923  [44800/60000]
loss: 1.152718  [51200/60000]
loss: 1.062558  [57600/60000]
Test Error: 
 Accuracy: 64.9%, Avg loss: 1.087809 

Done!
Saved PyTorch Model State to model.pth

Process finished with exit code -1
```