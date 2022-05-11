
## 1.构造数据
```python
data = np.arange(30).reshape(6, 5)
data
```

out:
```text
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14],
       [15, 16, 17, 18, 19],
       [20, 21, 22, 23, 24],
       [25, 26, 27, 28, 29]])
```
## 2.每次取出batch_size条数据
```python
print(data[[0, 1]])    
# 第一次: 0, 1        index : 0
# 第二次：2, 3        index : 1
# 第三次: 4, 5        index : 2
# 第四次：6, 7        index : 3
# 第五次: 8, 9        index : 4
# 故：开始下标 = 2*index 
print('---------------')
batch_size = 2   # 每次取出的样本个数
for index, record in enumerate(data):
    try:
        if index*batch_size + 1 < len(data):
            print(index)
            i = batch_size*index
            print(data[[i, i+1]])
#   输出结束时的判断：
        else:
            break
    except:
        print('index索引超过数组的最大索引!所有数据已经输出完')
        raise IndexError
```
out:
```python
[[0 1 2 3 4]
 [5 6 7 8 9]]
---------------
0
[[0 1 2 3 4]
 [5 6 7 8 9]]
1
[[10 11 12 13 14]
 [15 16 17 18 19]]
2
[[20 21 22 23 24]
 [25 26 27 28 29]]
```