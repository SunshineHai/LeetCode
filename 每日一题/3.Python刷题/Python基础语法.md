# Python基础

Python 是一种易于学习又功能强大的编程语言。它提供了高效的高级数据结构，还能简单有效地面向对象编程。Python 优雅的语法和动态类型，以及解释型语言的本质，使它成为多数平台上写脚本和快速开发应用的理想语言。

**Python的优点**：

- [ ] 语法简单，以缩进代替其它语言的大括号，适合阅读，使我们能够专注于解决问题而不是搞明白语言本身
- [ ] 易学。python虽然是用c语言写的，但是它摈弃了c中非常复杂的指针，简化了python的语法。
- [ ] 可移植性。由于它的开源本质，Python已经被移植在许多平台上（经过改动使它能够工作在不同平台上）。如果你小心地避免使用依赖于系统的特性，那么你的所有Python程序无需修改就可以在多种平台上面运行。
- [ ] Python既支持面向**过程**的函数编程也支持面向**对象**的抽象编程。
- [ ] 可扩展性和可嵌入性。如果你需要你的一段关键代码运行的更快或者希望某些算法不公开，你可以把你的部分程序用C或C++编写，然后在你的Python程序中使用它们。你可以把Python嵌入你的C/C++程序，从而向你的程序用户提供脚本功能。
- [ ] 丰富的库。Python标准库确实很庞大，第三方库也有很多方便的用处。比如：帮助你处理各种工作，包括正则表达式、文档生成、单元测试、线程、数据库、网页浏览器、CGI、FTP、电子邮件、XML、XML-RPC、HTML、WAV文件、密码系统、GUI（图形用户界面）和其他与系统有关的操作。

**缺点**：

- [ ] 运行速度慢。高级语言运行速度慢，Python对于C来说，就多了一个字节码转换成机器码的过程，所以相对会慢。但是人为是无法感知的。如果需要提高运行速度，可以通过嵌入C程序。
- [ ] 代码不能加密，因为PYTHON是解释性语言，它的源码都是以名文形式存放的，不过我不认为这算是一个缺点，如果你的项目要求源代码必须是加密的，那你一开始就不应该用Python来去实现。
- [ ] 线程不能利用多CPU问题，这是Python被人诟病最多的一个缺点，GIL即全局解释器锁（Global Interpreter Lock），是[计算机程序设计语言](http://zh.wikipedia.org/wiki/计算机程序设计语言)[解释器](http://zh.wikipedia.org/wiki/解释器)用于[同步](http://zh.wikipedia.org/wiki/同步)[线程](http://zh.wikipedia.org/wiki/线程)的工具，使得任何时刻仅有一个线程在执行，Python的线程是操作系统的原生线程。在Linux上为pthread，在Windows上为Win thread，完全由操作系统调度线程的执行。一个python解释器进程内有一条主线程，以及多条用户程序的执行线程。即使在多核CPU平台上，由于GIL的存在，所以禁止多线程的并行执行。

**Python 的特点**：

1. Python使用C语言开发，但是Python不再有C语言中的指针等复杂的数据类型。

2.  Python具有很强的面向对象特性，而且简化了面向对象的实现。它消除了保护类型、抽象类、接口等面向对象的元素。

3. Python代码块使用空格或制表符缩进的方式分隔代码。

4. Python仅有31个保留字，而且没有分号、begin、end等标记。

5. Python是强类型语言，变量创建后会对应一种数据类型，出现在统一表达式中的不同类型的变量需要做类型转换。

**Python的应用方向：**

　　1.常规软件开发

　　Python支持函数式编程和OOP面向对象编程，能够承担任何种类软件的开发工作，因此常规的软件开发、脚本编写、网络编程等都属于标配能力。

　　2.科学计算

　　随着NumPy，SciPy，Matplotlib，Enthoughtlibrarys等众多程序库的开发，Python越来越适合于做科学计算、绘制高质量的2D和3D图像。和科学计算领域最流行的商业软件Matlab相比，Python是一门通用的程序设计语言，比Matlab所采用的脚本语言的应用范围更广泛，有更多的程序库的支持。虽然Matlab中的许多高级功能和toolbox目前还是无法替代的，不过在日常的科研开发之中仍然有很多的工作是可以用Python代劳的。

　　3.自动化运维

　　这几乎是Python应用的自留地，作为运维工程师首选的编程语言，Python在自动化运维方面已经深入人心，比如Saltstack和Ansible都是大名鼎鼎的自动化平台。

　　4.云计算

　　开源云计算解决方案OpenStack就是基于Python开发的，搞云计算的同学都懂的。

　　5.WEB开发

　　基于Python的Web开发框架不要太多，比如耳熟能详的Django，还有Tornado，Flask。其中的Python+Django架构，应用范围非常广，开发速度非常快，学习门槛也很低，能够帮助你快速的搭建起可用的WEB服务。

　　6.网络爬虫

　　也称网络蜘蛛，是大数据行业获取数据的核心工具。没有网络爬虫自动地、不分昼夜地、高智能地在互联网上爬取免费的数据，那些大数据相关的公司恐怕要少四分之三。能够编写网络爬虫的编程语言有不少，但Python绝对是其中的主流之一，其Scripy爬虫框架应用非常广泛。

　　7.数据分析

　　在大量数据的基础上，结合科学计算、机器学习等技术，对数据进行清洗、去重、规格化和针对性的分析是大数据行业的基石。Python是数据分析的主流语言之一。

　　8.人工智能

　　Python在人工智能大范畴领域内的机器学习、神经网络、深度学习等方面都是主流的编程语言，得到广泛的支持和应用。

## 1. Python基础

知识点思维导图：

![image-20220221150656101](https://s2.loli.net/2022/02/21/HU5f4DczOIJkprM.png)

![image-20220221151042725](https://s2.loli.net/2022/02/21/6IGLDY5N8j9Vkcz.png)

### 1.1 环境搭建

Python 是一种易于学习又功能强大的编程语言。它提供了高效的高级数据结构，还能简单有效地面向对象编程。Python 优雅的语法和动态类型，以及解释型语言的本质，使它成为多数平台上写脚本和快速开发应用的理想语言。

Python是一门高级语言，是解释性语言，因此需要解释器，这里我们只需要下载Python解释器就可以执行Python语句。Python解释器可以在Python官网（ https://www.python.org/ ）中下载，访问的时候稍微有点慢。下载之后双击安装即可，安装时勾选安装环境即可。

注意：使用 Python解释器无法更换Python的版本，不够灵活，现在优先选择 Anaconda。

以下是百度百科的解释：

Anaconda指的是一个开源的[Python](https://baike.baidu.com/item/Python)发行版本，其包含了[conda](https://baike.baidu.com/item/conda/4500060)、Python等180多个科学包及其依赖项。 [1] 因为包含了大量的科学包，Anaconda 的下载文件比较大（约 531 MB），如果只需要某些包，或者需要节省带宽或[存储空间](https://baike.baidu.com/item/存储空间/10657950)，也可以使用**Miniconda**这个较小的发行版（仅包含conda和 Python）。

可以在官网下载Anaconda（https://www.anaconda.com/products/individual），也可以下载历史版本的Anaconda（https://repo.anaconda.com/archive/）。官网如果打开慢的话，可以使用国内的清华大学镜像网站：https://mirrors.tuna.tsinghua.edu.cn/anaconda/

安装好Anaconda之后可以在开始位置看到：

![image-20220221155821921](https://s2.loli.net/2022/02/21/VCjR95rHQtWpIGd.png)

打开Dos窗口，输入 conda -V 检测Anaconda是否安装成功：

conda 常用命令：

![image-20220223115949574](https://s2.loli.net/2022/02/23/RWMXOrmGEBf8yg7.png)

| 命令                            | 功能                     |
| ------------------------------- | ------------------------ |
| conda info -e                   | 查询有几个Python环境     |
| conda activate "环境名"         | 激活某个环境             |
| conda list/search               | 查看该环境下安装有哪些包 |
| conda install '包名'            | 安装指定包               |
| conda remove --name 环境名 包名 | 删除指定环境下的某个包   |
| conda remove 包名               | 删除当前环境下的包       |

具体可以参考如下博客：https://blog.csdn.net/y_h_k_666/article/details/118489382

### 1.2 变量

#### 1.2.1 定义变量

python中定义变量直接赋值即可，不需要声明是什么类型的，会自动识别，属于弱定义。例如：

```Python
a = 10 # 会自动把变量a赋值为整型类型的数组10
a
```

```python
string1 = 'hello' 	# 会自动把string1赋值为 字符串类型的变量
string2 = "China!"
string1, string2	# 字符串赋值时，使用单引号和双引号都可以
```

在 Python 中对一个变量（或者变量名）赋值时，你就创建了一个**指向**等号右边对象的**引用**。例如：我们定义一个整数列表时：

```python
a = [1, 2, 3]
```

假设我们将 a 赋值给一个新的变量 b :

```python
b = a
```

在 C 或者 Java 语言中，会是数据 [1, 2, 3] 被拷贝的过程。在 Python 中， a 和 b 实际上是指向了相同的对象，即原来的 [1, 2, 3]（如下图所示）。



我们可以在 a 中添加一个元素，然后检查 b 来证明。

```python
a = [1, 2, 3]
b = a

a.append(4)
b
```

结果：[1, 2, 3, 4]

问题：如何让赋值的时候实现重新拷贝一份？就是 和 C语言的 copy 一样。这里我们介绍一下Python中的**赋值**、**浅拷贝**和**深拷贝**，如下图所示：

![img](https://s2.loli.net/2022/02/23/HsCrOhuZiXLczlN.png)

1. 赋值：赋值引用，a 和 b 都指向同一个对象。

   ```python
   a = {1: [1, 2, 3]}  # 集合
   b = a
   print(a, id(a))
   print(b, id(b))
   # 子对象
   print(a[1], id(a[1]))
   print(b[1], id(b[1]))
   ```

   结果：

   ![image-20220223104041697](https://s2.loli.net/2022/02/23/sFL71R6Hh5dBrbN.png)

2. 浅拷贝，a 和 b 是一个独立的对象，但它们的子对象还是指向统一的对象（是引用）。如下所示：

   ```python
   a = {1: [1, 2, 3]}
   # b = a.copy()    # 浅拷贝
   b = copy.copy(a)
   
   print(a, id(a))
   print(b, id(b))
   # 子对象
   print(a[1], id(a[1]))
   print(b[1], id(b[1]))
   ```

   ![image-20220223104006365](https://s2.loli.net/2022/02/23/B6G8fTPweCaHkcs.png)



3. 深拷贝：b = copy.deep.copy(a)， a 和 b 完全拷贝了父对象及其子对象，两者是完全独立的。

   ```python
   import copy
   
   a = {1: [1, 2, 3]}
   b = copy.deepcopy(a)   # 深度拷贝
   
   print(a, id(a))
   print(b, id(b))
   
   # 子对象
   print(a[1], id(a[1]))
   print(b[1], id(b[1]))
   ```

   ![image-20220223104325083](https://s2.loli.net/2022/02/23/QirIq2zcHDpdAlE.png)

例如：

```python
a = 1
b = a
print(a, id(a)) # 返回对象的唯一标识符，标识符是一个整数。相当于对象的内存地址
print(b, id(b))
a = 2
print(a, id(a))
print(b, id(b))
```

结果：

```
1 140713741723536
1 140713741723536
2 140713741723568
1 140713741723536
```

![img](https://s2.loli.net/2022/02/23/GKDIYNA8ZSFh4QL.png)

例题：读程序，写结果。

```python
a = list(range(1, 7, 2))
b = a         # 注意 Python 里的赋值不是重新copy一份，而是把变量指向 该内存，一改都改，这样有利于节省空间。
a.pop(1)           
print(a)
b.append(6) 
print(a+b)    # 这里的 + 号，相当于是 把2个列表进行拼接 
print((a+b)[::-1])   # 倒序输出  
```

注意：

| 序号 | 方法名                                                       | 功能                                                         |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | [list.pop([index=-1\])](https://www.runoob.com/python/att-list-pop.html) | 根据索引移除列表中的一个元素（默认最后一个元素），并且返回该元素的值 |
| 2    | [list.append(obj)](https://www.runoob.com/python/att-list-append.html) | 在列表末尾添加新的对象                                       |

**Python列表脚本操作符**

列表对 + 和  * 的操作符与字符串相似。+ 号用于**组合列表**，* 号用于**重复列表**。

| Python 表达式                | 结果                         | 描述                 |
| ---------------------------- | ---------------------------- | -------------------- |
| len([1, 2, 3])               | 3                            | 长度                 |
| [1, 2, 3] + [4, 5, 6]        | [1, 2, 3, 4, 5, 6]           | 组合                 |
| ['Hi!'] * 4                  | ['Hi!', 'Hi!', 'Hi!', 'Hi!'] | 重复                 |
| 3 in [1, 2, 3]               | True                         | 元素是否存在于列表中 |
| for x in [1, 2, 3]: print x, | 1 2 3                        | 迭代                 |

**列表的切片**：截取

列表中的索引：与字符串的索引一样，列表索引从 0 开始，第二个索引是 1，依此类推。

![img](https://s2.loli.net/2022/02/23/Kldzm8qMf6OcX2F.png)

索引也可以从尾部开始，最后一个元素的索引为 -1，往前一位为 -2，以此类推。

![img](https://s2.loli.net/2022/02/23/SMzyYRNoauKUxcG.png)

| Python 表达式 | 结果                  | 描述                                                         |
| ------------- | --------------------- | ------------------------------------------------------------ |
| L[2]          | 'Taobao'              | 读取列表中第三个元素                                         |
| L[-2]         | 'Runoob'              | 读取列表中倒数第二个元素                                     |
| L[1:]         | ['Runoob', 'Taobao']  | 从第二个元素开始截取列表                                     |
| L[1:-1]       | [1, 2, 3, 4, 5, 6]    | 从第2个元素开始截取，到最后一个元素，不包含租后一个元素      |
| L[: : -1]     | [0,,1, 2, 3, 4, 5, 6] | a[: : n] 如果 n 为正数，等价于 a[0: len(a)+1: n]， 如果 n 为负数(从右边开始截取)，等价于 a[-1: -len(a)-1: -1] |

```python
a = [0, 1, 2, 3, 4, 5, 6]
print(a[::2], a[0: 7: 2]) # 等价于  a[0, 7, 2]
# a[: : n] 如果 n 为正数，等价于 a[0: len(a)+1: n]， 如果 n 为负数(从右边开始截取)，等价于 a[-1: -len(a)-1: -1]
a[: : -1], a[-1: -len(a)-1: -1]
```

结果：![image-20220223145253856](https://s2.loli.net/2022/02/23/jgleKyYuNVLI3Q1.png)

#### 1.2.2 关键字

以下标识符为保留字，或者称为**关键字**，不可用于普通标识符。关键字的拼写必须与这里列出的完全一致：

```python
False      await      else       import     pass
None       break      except     in         raise
True       class      finally    is         return
and        continue   for        lambda     try
as         def        from       nonlocal   while
assert     del        global     not        with
async      elif       if         or         yield
```

#### 1.2.3 命名规则

​         PEP8风格指南：

- [ ] 函数、变量及属性应该用小写字母来拼写，各单词之间以下划线相连，例如：lowercase_undersome

  ```python
  def get_max(a:int, b:int):  # 函数名称的命名
      if a > b:
          return a
      else:
          return b
  get_max(20, 10)
  ```

  ```python
  def get_max(a:int, b:int):
      # 下面这句话类似于C语言里的三目运算符
      return a if a > b else b
  get_max(10, 20)
  ```

- [ ]   受保护的实例属性，应该以单个下划线开头，例如，_leading_underscore

- [ ]   私有的实例属性，应该以两个下划线开头，例如，__double_leading_underscore。

- [ ]   类与异常，应该以每个单词首字母均大写的形式来命名，例如：CapitalizedWord。

- [ ]   模块级别的常量，应该全部采用大写字母来拼写，各单词之间以下划线相连，例如：ALL_CAPS

- [ ]   类中的实例方法（instance method），应该把首个参数命名为 self，以表示该对象本身。   

  ```python
  class Complex:
      def __init__(self, realpart, imagpart):
          self.r = realpart
          self.i = imagpart
      
      def f(self):
          return 'Hello World!'
  x = Complex(3.0, -4.5)   # 对象初始化
  print(x.f())
  x.r, x.i
  # 结果：
  # Hello World! 
  # (3.0, -4.5)
  ```

- [ ]  类方法（class method）的首个参数，应该命名为 cls ，以表示该类自身。           

  注意：类方法和实例方法类似，它最少也要包含一个参数，只不过类方法中通常将其命名为cls，Python会自动将类本身绑定给cls参数（注意，绑定的不是类对象）。也就是说，我们在调用类方法时，无需显式为cls参数传参。

  ​	和实例方法最大的不同在于，类方法需要使用**@classmethod**修饰符进行修饰，例如：

  ```python
  class language:
      # 定义构造方法， 也属于实例方法
      def __init__(self):
          self.name = '我是Python语言'
          self.add = '可要认真学习哦'
          
      # 定义实例方法：使用类的对象进行调用
      def say(self):
          print("正在调用say()实例方法")
          
      # 定义 类方法
      @classmethod
      def info(cls):
          print("正在调用类方法", cls)
  ```

  类方法推荐使用类名直接调用，当然也可以使用实例对象来调用（**不推荐**）
  
  ```python
  # 使用类名直接调用类方法：
  language.info()
  # 使用类的对象调用类方法
  lan = language()
  lan.info()
  
  # 调用实例方法：
lan.say()
  ```

  执行结果：
  
  ```python
  正在调用类方法 <class '__main__.language'>
  正在调用类方法 <class '__main__.language'>
正在调用say()实例方法
  ```
  
  ​                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            

#### 1.2.4 基本数据类型

Python中定义有五种标准的数据类型：

- [ ] Numbers(数字)

  其下有四种不同的数字类型：

  - [ ] int(有符号整型)

  - [ ] long(长整型，也可以代表八进制和十六进制) 

  - [ ] float(浮点型)

  - [ ] complex(复数)

    注意：long型只*存在于 Python2.X 版本中*，*在 Python3.X 版本中 long 类型被移除，使用 int 替代。*

- [ ] String(字符串)

- [x] List(列表)

- [x] Tuple(元组)

- [x] Dictionary(字典)

1. Numbers 类型

```python
a = 10
a
```

![image-20220227145202146](C:\Users\JackYang\AppData\Roaming\Typora\typora-user-images\image-20220227145202146.png)

```python
# del ： 删除单个或多个对象的引用
del a
a
```

![image-20220227145241493](C:\Users\JackYang\AppData\Roaming\Typora\typora-user-images\image-20220227145241493.png)

2. 字符串

![image-20220227145844173](https://s2.loli.net/2022/02/27/rPCG346zuq7toMp.png)

索引：

正向索引：从左到右索引默认0开始的，最大范围是字符串长度少1，和C语言中的数组下标一致。

负向索引：从右到左索引默认-1开始的，最大范围是字符串开头。这一点是Python特有的。

正/负向索引如下图所示：

![img](https://s2.loli.net/2022/02/27/dfos3wZyBSg4iTv.png)

切片：

字符串的切片就是根据自己的需求取字符串的字串，格式如下：变量名[头下标： 尾下标 ： 步长]，从头下标开始截取，到尾下标结束（不包含尾下标），间隔为步长。即以步长为单位，**左闭右开**。

```python
letter = 'I love China!'
# 使用正向索引，截取China字符：
print(letter[7:12])  
# 使用负向索引截取：
letter[-6:-1]
```

![image-20220227151352428](https://s2.loli.net/2022/02/27/tXRwBWiKFQ1mnlE.png)

```python
# 变量名[头下标：尾下标：步长]：使用切片时，如果头下标、尾下标或者步长有缺失时：
print(letter[0:13:2]) 
print(letter[0:13])
print(letter[:13])
print(letter[2:])
print(letter[:-1])
print(letter[-1:-13:-1]) # 步长为负，表示索引依次减少；步长为正表示索引依次增加
# 原样输出：
print(letter[:len(letter)])
print("简写：", letter[::1])
# 逆向输出：
print(letter[-1:-len(letter):-1])
letter[::-1]  # 等价于前一句
```

![image-20220227152529448](https://s2.loli.net/2022/02/27/ZkPdHWGRvuByQ1f.png)

3. 列表

定义时使用 [] 表示，例如 ：

```python
t = ['a', 'b', 'c', 'd', 'e']
```

列表中的子对象可以改变，是可改变类型。例如：修改列表中的‘c’为'C':

```python
t[2] = 'C'
t
```

结果：![image-20220227153728367](https://s2.loli.net/2022/02/27/mLiTH3yVsuEobP5.png)

列表的切片和字符串的切片一样，这里不再重复解释。

![img](https://s2.loli.net/2022/02/27/Pha9jmUyovR7qC5.png)

注意：+ 号是列表**连接**运算符，\* 是重复操作

```python
# 列表：
my_list = ['google', 'baidu', 'alibaba']
print(my_list * 2) # 输出列表两次
sub_list = ['520']
my_list + sub_list  # 列表的拼接，不是算术上的相加 
```

![image-20220227153100699](https://s2.loli.net/2022/02/27/QW4Gh2xBztnLNiE.png)

4. 元组

元组是另外一种数据类型，内部元素以逗号隔开。但是元组不能二次赋值，是不可变对象。

```python
# 元组
my_tuple = ('baidu', 520, 3.14, 'Jack', ['I', 'love', 'China'])
my_tuple[0] = "百度"  # 需求改变元组中的第一个元素时报错
my_tuple
```

![image-20220227154238747](https://s2.loli.net/2022/02/27/h3j4gN7k8K5tTwd.png)

注意以下这样使用是可以的：![image-20220227154428572](https://s2.loli.net/2022/02/27/9RnHSflBFXuJszO.png)

虽然元组是不可变对象，但是元组里面的列表是可变对象，因此我们可以进行修改。

5. 字典

   字典（dictionary）是除列表以外python之中最灵活的内置数据结构类型。列表是有序的对象集合，字典是无序的对象集合。

   字典的定义：通过键值对来存储对象。使用 “{key : value}” 来定义， 由索引 key 和 对应的value 组成。

#### 1.2.5 常用的数据结构

- [ ] 元组（Tuple）
- [ ] 列表（list）
- [ ] 字典（dictionary）
- [ ] 集合（set）

##### 1.2.5.1 元组

```python
tup = 4, 5, 6  # 创建元组等价于： tup = (4, 5, 6)
tup
```

![image-20220227155722699](https://s2.loli.net/2022/02/27/f4oOr5JQmkw83Va.png)



1.  "+" 和 "\*" 用法和在列表中的用法一致。

```python
# + ： 连接功能
# * ： 复制功能
(4, None, 'foo') + (6, 0) + ('bar', )  # 元组中只有一个字符串时，右边需要加一个逗号，防止识别时和单个字符串混淆
```

![image-20220227160332059](https://s2.loli.net/2022/02/27/UfoEFnb7DJAP4tO.png)

2. 元组拆包：

当把一个元组赋值给一个变量时，右边的元组就会把右边的值进行拆包。

```python
a, b, c = (4, 5, 6)
b
```

![image-20220227160659711](https://s2.loli.net/2022/02/27/ZsrJ2HcMmERlieY.png)

交换2个元素的值，Python语言可以这样写：

```python
a, b = 1, 2
print(a, b)
a,b = b, a
print(a, b)
```

![image-20220227160912284](https://s2.loli.net/2022/02/27/Bvhq5wkiCFJrZ7m.png)

遍历列表时进行拆包：

```python
seq = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
for a,b,c in seq:     # 遍历 列表seq, 然后把每个元组赋值给3个变量
    print(a, b, c)
```

![image-20220227161608206](https://s2.loli.net/2022/02/27/j5FTNKGZRHhJUVW.png)

3. 元组方法

   count(形参) ： 计算某变量在元组中出现的次数。

   ```python
   a = (1, 2, 2, 2, 3, 4, 2)
   a.count(2)
   ```

   ![image-20220227161902921](https://s2.loli.net/2022/02/27/5Ufmo7s4hSVtkIQ.png)

##### 1.2.5.2 列表

1. 定义

```python
# 列表
# 1.定义：
a_list = [2, 3, 7, None]
a_list

tup = ('apple', 'banana')
b_list = list(tup)
b_list
```



```python
# list() 长用于将迭代去转化为列表
gen = range(10)
print(gen)
list(gen)
```

![image-20220228171858206](https://s2.loli.net/2022/02/28/j371PtJ5NcVvMmH.png)

2. 增加或移除元素

   | 方法       | 含义                                                  |
   | ---------- | ----------------------------------------------------- |
   | appen(obj) | 将元素/对象添加到列表的尾部。                         |
   | pop(index) | 移除**索引为index**的元素，形参为空则移除最后一个元素 |

   测试如下：

   ```python
   print(b_list)
   b_list.append('orange')
   print(b_list)
   print(b_list)
   b_list.append('orange')
   print(b_list)
   ```

   ![image-20220228172735785](https://s2.loli.net/2022/02/28/QcITkmwKbOj9hJR.png)

   

   ```python
   c_list = ['a', 'b', 'c']
   print(b_list)
   b_list.append(c_list)
   print(b_list)
   ```

   ![image-20220228172744432](https://s2.loli.net/2022/02/28/CObtVpsKqfNaAIX.png)



#### 1.2.5 类型转换

### 1.3 运算符和表达式

### 1.4 流程控制

### 1.5 基本数据类型

### 1.6 函数

### 1.7 面向对象编程

### 1.8 文件操作



# NumPy知识点


```python
X = np.array([[9, 8, 7, 2], [4, 6, 5, 7], [9, 6, 1, 3], [20, 10, 12, 16]]) # 定义二维numpy数组
X
```


    <IPython.core.display.Javascript object>
    
    array([[ 9,  8,  7,  2],
           [ 4,  6,  5,  7],
           [ 9,  6,  1,  3],
           [20, 10, 12, 16]])


```python
X[X>10] # 找到 X[0]~X[3] 中 每组值都大于 10 的
```




    array([20, 12, 16])




```python
X > 10 # 这条语句会对 二维数组 X 中的每个元素进行比较 > 10 的为 True， 否则为 False.
```




    array([[False, False, False, False],
           [False, False, False, False],
           [False, False, False, False],
           [ True, False,  True,  True]])

# pandas 知识点


```python
import pandas as pd

products = pd.DataFrame({
                            'product' :  ['apple', 'banana', 'orange'],
                            'date' :     ['2021-12-20', '2022-1-6', '2022-3-8'],
                            'quantities':[12, 8, 30]

})
products
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product</th>
      <th>date</th>
      <th>quantities</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>apple</td>
      <td>2021-12-20</td>
      <td>12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>banana</td>
      <td>2022-1-6</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>orange</td>
      <td>2022-3-8</td>
      <td>30</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 查找 products 表中，金额最大的列
qmax = products['quantities'].max()  # 求 quantities 列的最大值
qmax
```




    30




```python
row = products.iloc[:]
row
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product</th>
      <th>date</th>
      <th>quantities</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>apple</td>
      <td>2021-12-20</td>
      <td>12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>banana</td>
      <td>2022-1-6</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>orange</td>
      <td>2022-3-8</td>
      <td>30</td>
    </tr>
  </tbody>
</table>
</div>




```python
products.index()
```


    ---------------------------------------------------------------------------
    
    TypeError                                 Traceback (most recent call last)
    
    <ipython-input-104-89fa29b2384f> in <module>
    ----> 1 products.index()


    TypeError: 'RangeIndex' object is not callable


## pandas.iloc[] 的使用 

### 索引一个轴得到行记录


```python
mydict = [{'a': 1, 'b': 2, 'c': 3, 'd': 4},
          {'a': 100, 'b': 200, 'c': 300, 'd': 400},
          {'a': 1000, 'b': 2000, 'c': 3000, 'd': 4000 }]
df = pd.DataFrame(mydict)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100</td>
      <td>200</td>
      <td>300</td>
      <td>400</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1000</td>
      <td>2000</td>
      <td>3000</td>
      <td>4000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.iloc[1]   #  类型：pandas.core.series.Series, 结果是 Series 类型, 以列的形式展开

```




    a    100
    b    200
    c    300
    d    400
    Name: 1, dtype: int64




```python
df.iloc[[0]]  # 第一行的记录
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
type(df.iloc[[0]])
```




    pandas.core.frame.DataFrame




```python
df.iloc[[0, 1]] # 第一行和第二行的记录
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100</td>
      <td>200</td>
      <td>300</td>
      <td>400</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.iloc[:3] # 前3行的记录，索引是从零开始
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100</td>
      <td>200</td>
      <td>300</td>
      <td>400</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1000</td>
      <td>2000</td>
      <td>3000</td>
      <td>4000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.iloc[[True, False, True]] # 第一行和第三行的数据
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1000</td>
      <td>2000</td>
      <td>3000</td>
      <td>4000</td>
    </tr>
  </tbody>
</table>
</div>



### 索引2个轴得到一个数


```python
df.iloc[0, 1]  # 得到 第 0 行 1 列的数，行、列索引从0开始
```




    2




```python
df.iloc[[0, 2], [1, 3]] # 第 0 行 和 第1行的数据， 第 1列和第3列的数据，得到第0行1列 和 第2行3列 的2个数
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>b</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2000</td>
      <td>4000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 使用切片对象
df.iloc[1:3, 0:3]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>100</td>
      <td>200</td>
      <td>300</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1000</td>
      <td>2000</td>
      <td>3000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.iloc[0:, 0:] # df.iloc[行, 列] 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100</td>
      <td>200</td>
      <td>300</td>
      <td>400</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1000</td>
      <td>2000</td>
      <td>3000</td>
      <td>4000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.iloc[:, lambda df: [0, 2]] # 使用 lambda 匿名函数
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100</td>
      <td>300</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1000</td>
      <td>3000</td>
    </tr>
  </tbody>
</table>
</div>



## pandas.loc[]的使用

## 获取值


```python
# 二维的 DataFrame 对象，和excel的表格数据类似，行标签用index表示，列标签用columns表示
df = pd.DataFrame([[1, 2], [4, 5], [7, 8]],
     index=['cobra', 'viper', 'sidewinder'],
     columns=['max_speed', 'shield'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>max_speed</th>
      <th>shield</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>cobra</th>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>viper</th>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>sidewinder</th>
      <td>7</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc['viper']
```




    max_speed    4
    shield       5
    Name: viper, dtype: int64




```python
type(df.loc['viper'])  # 行标签
```




    pandas.core.series.Series




```python
df.loc[['viper', 'sidewinder']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>max_speed</th>
      <th>shield</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>viper</th>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>sidewinder</th>
      <td>7</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc['cobra', 'shield'] # 得到 行列 对应的值
```




    2




```python
type(df.loc['cobra', 'shield'])
```




    numpy.int64




```python
df.loc['cobra':'viper', 'max_speed']
```




    cobra    1
    viper    4
    Name: max_speed, dtype: int64
