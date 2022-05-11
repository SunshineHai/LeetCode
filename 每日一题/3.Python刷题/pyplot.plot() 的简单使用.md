# 画图

pyplot.plot() 的官方文档解释：[链接](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html?highlight=pyplot%20plot#matplotlib.pyplot.plot)

## 1.1 画二维平面图

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 10, 3)
y = np.arange(0, 10, 3)
print(x)
plt.plot(x, y, '*g--')
plt.show()
```

![](https://s2.loli.net/2022/04/05/5L3fy9omCRI4pGx.png)

以上代码是画 二维平面图，```python plt.plot(x, y, '*g--')``` 这里的x 表示横坐标， y 表示纵坐标， 字符串 ```python '*g--' ``` 其中 \* 表示每个点用 五角星表示， g 表示线段的颜色为绿色， -- 表示线段的类型，如果需要更换 点的形状、 颜色、线段可以参考官方文档。

## 1.2 同一个坐标系里画多条线段

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 10, 3)
y1 = x
y2 = x**2

plt.plot(x, y1, '*g--', y2, '^b-')
plt.legend(['y=x', '$y=x^2$'], loc='upper right')  # 显示每条线段的解释， $$ 里是 LaTeX语句
plt.show()
```

![](https://s2.loli.net/2022/04/05/n3y5aGzRiWPjCpd.png)

注意：默认如果使用汉字，显示是不正常的，这里重新设置显示字体即可，代码如下：

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 10, 3)
y1 = x
y2 = x**2

# 正常显示中文
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']

plt.xlabel('自变量')   # 若是使用 汉字，则显示出错
plt.ylabel('因变量')
plt.plot(x, y1, '*g--', y2, '^b-')
plt.legend(['y=x', '$y=x^2$'], loc='upper right')  # 显示每条线段的解释， $$ 里是 LaTeX语句
plt.show()
```

## 1.3 封装画图的函数

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def show_error(x_name:list, precision:list, recall:list, f1:list, AUC:list, path):
    plt.plot(x_name, precision, 'or-', recall, '^g-', f1, '*b-', AUC, '.y-.')
    plt.legend(['precision', 'recall', 'f1', 'AUC'], loc='upper right')
    plt.savefig(path, dpi=300)
    plt.show()
    pass

x_name = ['model1', 'model2', 'model3', 'model4']

y1 = [0.5, 0.6, 0.8, 0.9]
y2 = [0.6, 0.6, 0.85, 0.92]
y3 = [0.7, 0.66, 0.82, 0.91]
y4 = [0.8, 0.7, 0.82, 0.84, 0.93]
show_error(x_name,y1 ,y2 ,y3 , y4, r'test.png')
```

![](https://s2.loli.net/2022/04/05/bTkQX9ANnpPY5mR.png)

这里每一条线段代表一个**评价指标**, 横坐标代表不同的模型，纵坐标代表指标的大小，每一条折线，代表该不同模型的相同评价指标的值。通过折线图可以直观的观察出模型的好坏。