## 1.第一步:要找到配置文件位置
```text
jupyter notebook --generate-config
```

![](https://s2.loli.net/2022/03/28/3ENljrTLx9swDqR.png)

```text
 C:\Users\JackYang\.jupyter\jupyter_notebook_config.py
```

## 第二步：打开配置文件
搜索  c.NotebookApp.notebook_dir = ''
把要打开的路径赋值到上面。