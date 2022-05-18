
#  1.使用python 内建的open()方法读取文本

        相对路径：example/ex2.txt，文件内容如下所示：

![](https://img-blog.csdnimg.cn/2021070711031826.png)![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")​编辑

 测试内容，路径和内容，大家可根据自己心情设置。

使用open()方法读取：

```
print('----使用 python自带的open() 读取文件-----')
path = r'example/ex2.txt'
frame = open(path)
print(frame.readlines())
```

![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")

 此时，执行结果报错如下：

![](https://img-blog.csdnimg.cn/20210707110448937.png)![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")​编辑

 我猜测open() 方法的默认编码不支持中文读取，假如 我把TXT 文件中的汉语删除，再次执行：

![](https://img-blog.csdnimg.cn/20210707110628927.png)![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")​编辑

success！但是如何输出汉字哪？我猜测手动指定open（）方法解析文本的编码方式 ，增加 encoding='utf-8'。

```
path = r'D:\PythonTest\20200925\example\ex2.txt'
frame = open(path, encoding='utf-8')
print(frame.readlines())
frame.close()# 不用则把文件关闭
```

![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")

 ![](https://img-blog.csdnimg.cn/20210707111029990.png)![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")​编辑

 完美读取出来！

不加会报错：

![](https://img-blog.csdnimg.cn/20210707111102715.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3lfaF9rXzY2Ng==,size_16,color_FFFFFF,t_70)![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")​编辑

# 2.使用 pandas 读取

使用 ExcelFile ，通过将 xls 或者 xlsx 路径传入，生成一个实例。

```python
import pandas as pd

xlsx = pd.ExcelFile(r'example/ex1.xlsx')
print(xlsx)
print(type(xlsx))

print(pd.read_excel(xlsx, 'Sheet1'))
```

![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")

 Excel 的表格内容如下：

![](https://img-blog.csdnimg.cn/20210707102847661.png)![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")​编辑

 此时报错：

![](https://img-blog.csdnimg.cn/20210707102950354.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3lfaF9rXzY2Ng==,size_16,color_FFFFFF,t_70)![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")​编辑

 注意：读取 后缀名为 ‘.xlsx’ 的Excel文件，需要使用附加包 'xlrd' (读取 .xls)和 ‘openpyxl’(读取 .xlsx)，于是我就根据报错提示安装：

```
conda install xlrd
```

![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")

 安装结果：

![](https://img-blog.csdnimg.cn/20210707103252188.png)![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")​编辑

 之后执行代码 依然报错：

![](https://img-blog.csdnimg.cn/20210707103327536.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3lfaF9rXzY2Ng==,size_16,color_FFFFFF,t_70)![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")​编辑

 依然不支持读取。这时，我们再安装 ‘openpyxl’ 包：

```
conda install openpyxl
```

![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")

 此时 依然报错：

![](https://img-blog.csdnimg.cn/20210707103836913.png)![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")​编辑

方法一：使用 engine='openpyxl' 读取 Excel文件。

```
import pandas as pd

# 使用 ExcelFile ，通过将 xls 或者 xlsx 路径传入，生成一个实例
xlsx = pd.ExcelFile(r'example/ex1.xlsx' , engine='openpyxl') #
print(type(xlsx))
print(xlsx)
print(type(xlsx))
```

![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")

此时可以正常读取文件表格，终于成功了：

![](https://img-blog.csdnimg.cn/20210707104527518.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3lfaF9rXzY2Ng==,size_16,color_FFFFFF,t_70)![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")​编辑

法二：Package xlrd 默认安装的版本如下

![](https://img-blog.csdnimg.cn/20210707104659156.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3lfaF9rXzY2Ng==,size_16,color_FFFFFF,t_70)![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")​编辑

 引用自 [pandas无法打开.xlsx文件，xlrd.biffh.XLRDError: Excel xlsx file； not supported_氦合氢离子的博客-CSDN博客](https://blog.csdn.net/weixin_44073728/article/details/111054157 "pandas无法打开.xlsx文件，xlrd.biffh.XLRDError: Excel xlsx file； not supported_氦合氢离子的博客-CSDN博客") 来源网络，如有侵权联系删除。

更换 xlrd 的版本为 1.2.0。

# ![](https://img-blog.csdnimg.cn/20210707104924506.png)![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")​编辑

执行一下代码：

```
import pandas as pd

# 使用 ExcelFile ，通过将 xls 或者 xlsx 路径传入，生成一个实例
xlsx = pd.ExcelFile(r'example/ex1.xlsx') #
print(type(xlsx))
print(xlsx)
print(type(xlsx))
```

![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")

 成功读取Excel 表格。

# 3.使用 pandas读取的简单方法

经过上一步的麻烦设置，我们不在理睬这2个包，开始尽情的使用python操作Excel表格。

直接使用 read_excel() 读取表格。

code如下，方便copy

```
import pandas as pd
path = r'D:\PythonTest\20200925\example\ex1.xlsx'
frame = pd.read_excel(path)   # 直接使用 read_excel() 方法读取
frame
```

![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")

![](https://img-blog.csdnimg.cn/20210707105525907.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3lfaF9rXzY2Ng==,size_16,color_FFFFFF,t_70)![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")

  
