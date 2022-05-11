# 数据结构与算法(Python语言描述)P86和P88判断谓词和谓词参数的理解

## 1.书中说的"判断谓词"和"谓词参数"

P86页代码如下：

```python
def find(self, pred):
    p = self._head
    while p is not None:
        if pred(p.elem):
            return p.elem
        p = p.next
```

这里的 pred 是一个函数，相当于形参里参入一个实现判断功能的函数，对于Python语言传入函数名即可。理解如下：

假如我们定义一个find函数如下：

```python
def find(num:[list], pred):
    for x in num:
        if pred(x):
            return x
    pass
```

该函数相当于把 num 进行遍历，num里的每一个元素都调用 pred()。

```python
def equals(elem):
    return elem == 1
```

定义 equals(elem) 判断elem是否等于1，等于1则返回True，否则返回False。调用find()函数：

```python
res = find([9, 1, 2, 3, 1, 2, 3], equals)
print(res)
```

![image-20220318090807033](https://s2.loli.net/2022/03/18/K6YHSMLklEFBJTv.png)

结果是返回num列表里第一个与1相当的数。

equal()方法可以使用匿名函数(lambda表达式)实现：

```python
res = find(num, lambda x: x == 1)
print("Use lambda:{0}".format(res))
```

![image-20220318091100844](https://s2.loli.net/2022/03/18/NcVrbzxdYP6tCSg.png)

## 2.使用生成器

上边找到的是num里与1相等的第1个元素的值，假如要找到num里与1相等的所有的值，这里我们使用生成器函数，把满足条件的放到生成器里面，然后进行迭代：

```python
def filter(num:[list], pred):  # 生成器函数
    for x in num:
        if pred(x):
            yield x
    pass
# 迭代器
print('----------------------------')
for x in filter(num, lambda x: x == 1):
    print(x)
```

![image-20220318093211819](https://s2.loli.net/2022/03/18/9reqos2COb5Kzak.png)

## 3.使用C语言测试

由于C语言里没有匿名函数，所以我们使用 形参里使用函数名：

```c
#include<stdio.h>

int equals(int elem){
	return elem == 1;
}

int find(int a[], int len, int (*fun)(int x) ){   //指向函数的指针 
	int i;
	for(i = 0; i < len; i++){
		if(fun(a[i])) 
			return a[i];
	}
}

void main(){
	int a[] = {1, 2, 3, 4, 5, 6};
	int len = 6;
	printf("len = %d\n", len); 
	int res = find(a, len, equals);
	printf("res = %d\n", res);
} 

```

**注意：**

```c
int (*fun)(int x)
    
find(a, len, equals);
```

这里是指向函数的指针，相当于函数作为形参，调用时实参使用函数名即可。

个人理解，如有不对的地方，请大家指正，谢谢。

## 4. 测试

- 无序列表1
- 无序列表2
- 无序列表3

一级引用如下：

> 一级引用示例：
>
> 读一本好书就是和高尚的人说话。	**----歌德**



---

Use `printf()` function. 

 ```diff
+ 新增项
- 删除项
 ```

> 支持平台：微信公众号。

通过`<![](url),![](url)>`这种语法设置横屏滑动滑动片，具体用法如下：

<![蓝1](https://files.mdnice.com/blue.jpg),![绿2](https://files.mdnice.com/green.jpg),![红3](https://files.mdnice.com/red.jpg)>



::: block-1
### 容器块 1 示例

> 读一本好书，就是在和高尚的人谈话。 **——歌德**
> :::