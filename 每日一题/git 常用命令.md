## 1. Git 常用命令

 git  induction:

![img](https://i.loli.net/2021/07/20/W9RvZpAiYmxQf7c.jpg) 

工作区 --> 暂存区-->本地仓库-->远程仓库

一般现在github上建立远程仓库，然后再clone到本地，这样不容易出错。

```git
git clone + 远程仓库地址
```

### 1.1 添加文件到暂存区

1. 添加单个文件到暂存区
```git
git add 文件名    
```

2.添加所有文件到暂存区
```
git add .
```

### 1.2 提交文件到本地仓库
1. 提交单个文件到本地仓库

```
git commit 文件名 -m '注释'
```

2.提交所有修改过的文件到本地仓库
```
git commit -m '注释'
```
### 1.3 提交文件到远程仓库

```
命令格式：
git push <远程主机名> <本地分支名>:<远程分支名>
如果本地分支名与远程分支名相同，则可以省略冒号：
git push <远程主机名> <本地分支名>
git push origin main
```

### 1.4 commit 的文件>100M

​	提交的文件超过100M，则会报错，这是该怎么办？

- 安装lfs

```
git lfs install
```

- 选择需要提交的git lfs 文件

```
git lfs track "需要提交的文件"
```

- 跟踪 .gittattributes

```
git add .gittattributes
```

- 测试