

         当你使用Latex公式后，你会悄悄的爱上它。

LaTex 百度百科解释：

LaTeX（LATEX，音译“拉泰赫”）是一种基于ΤΕΧ的[排版](https://baike.baidu.com/item/%E6%8E%92%E7%89%88/1824351 "排版")系统，由[美国](https://baike.baidu.com/item/%E7%BE%8E%E5%9B%BD/125486 "美国")计算机学家莱斯利·兰伯特（Leslie Lamport）在20世纪80年代初期开发，利用这种格式，即使使用者没有排版和程序设计的知识也可以充分发挥由[TeX](https://baike.baidu.com/item/TeX/3794463 "TeX")所提供的强大功能，能在几天、甚至几小时内生成很多具有书籍质量的印刷品。对于生成复杂表格和[数学公式](https://baike.baidu.com/item/%E6%95%B0%E5%AD%A6%E5%85%AC%E5%BC%8F/10349953 "数学公式")，这一点表现得尤为突出。因此它非常适用于生成高印刷质量的科技和数学类文档。这个系统同样适用于生成从简单的信件到完整书籍的所有其他种类的文档。

## 1.1 LaTex 基本框架

         LaTex 分为导言区和正文区，简单格式如下：

```
\documentclass[UTF8]{ctexart}	% [UTF-8] ：文档编码 {ctexart} : 中英混输模式。

% '%'是注释符号；这里是导言区
% 导言区：不显示内容，用来引用一些宏包、设置页面格式等。

\begin{document}

% 正文区 ： 用来编辑正文，包括标题、段落、生成目录、插入图片、插入公式等
庆祝中国共产党成立一百周年！

Celebrating the centenary of the founding of the Communist Party of China!

\end{document}
```

![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")

运行结果：

![](https://img-blog.csdnimg.cn/2021070508351631.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3lfaF9rXzY2Ng==,size_16,color_FFFFFF,t_70)![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")​编辑

##  1.2 Latex 的简单使用

        LaTex 主要用于论文的排版，可以设置标题、生成目录、插入图片、插入有序、无序列表、插入公式等。其中我最喜欢的是插入公式这个功能，真是太方便了，比word节省了好多时间而且还插入的公式还特别好看。例子如下：

```
% 默认模式 ： {article} 
% 中英混输模式：{}
\documentclass[11pt,a4paper]{article} % 设置A4纸，字体大小
% 导言区
\usepackage{ctex} % 使用中文包

\title{LaTeX快速入门}
\author{JackYang}
\date{\today}

\usepackage{graphicx} % 插入图片的包

\usepackage{amsmath}	% 编写数学公式使用的包
% 正文区(文稿区)
\begin{document}
	 \maketitle   % 使用该语句之后，标题会显示处出来
	 \tableofcontents % 目录
	 Hello World! 我是中国人！
你好！LaTeX\footnote{LaTeX是一个与Word比肩，甚至更好的工具}  % 脚注

% 标题级别

\part{part标题}
\section{一级标题}
\subsection{subsection标题}
\subsubsection{subsubsection标题}
\paragraph{paragraph标题}
\subparagraph{subparagraph标题}

% 插入图片

\begin{figure}[h]
\begin{center}
\includegraphics[scale=0.2]{latex.jpg}
\end{center}
\caption{该图显示了一个人的测试示例。 它表明我们的系统跟踪人进入房间时的姿势，甚至当他完全被遮挡在墙后时。}
\label{fig:latex}
\end{figure}

% 无序列表
\begin{itemize}
\item \textbf{上图}：由与无线电传感器共同定位的相机拍摄的图像，并在此处显示以供视觉参考。
\item \textbf{中间}：仅从RF信号中提取的关键点置信度图，没有任何视觉输入。
\item \textbf{底部}：从关键点置信度图解析的骨架，表明即使存在完全遮挡，我们也可以使用RF信号来估计人体姿势。
\end{itemize}

% 公式
% 行内公式
\section{公式的使用} 
\subsection{行内公式}
大家好，我是$a^2+b^=c^2$行内公式。
\subsection{行间公式}
勾股定理的公式如下所示：
\[
	a ^ 2 + b^2 = c^2
\]
\subsection{行间公式自动编号}
\begin{equation}
	a ^ 2 + b^2 = c^2
\end{equation}

\begin{equation}
	1+2+3+\dots+(n-1)+n = \frac{n(n+1)}{2}
\end{equation}

\begin{equation}
	MAE = \frac{1}{n} \sum _{i=1} ^{n} \left| \widetilde{y} _{i} - y _{i} \right|
\end{equation}

\begin{equation}
	RMSE = \sqrt{\frac{\sum \limits _{i=1} ^n(y_i - \widetilde{y}_i)^2}{n-1}}
\end{equation}


\begin{equation}
	R^2 = 1 - \frac{\sum \limits _{i=1} ^n(y_i - \widetilde{y}_i)^2}{\sum \limits _{i=1} ^n(y_i - \overline{y}_i)}
\end{equation}

\end{document}

```

![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")

## 1.3 LaTex 公式

        我主要给大家分享下 LaTex 公式的使用。LaTex公式分为 行内公式、行间公式和行间公式自动标号。以下内容，容我详细聊聊：

### 1.3.1 行内公式

        语法：使用两个美元符号，两个美元符号中间写公式内容。相当于在文中中间插入公式或者数学符号。

```
$ 公式内容 $
% 举列子如下：
% 行内公式
\section{公式的使用} 
\subsection{行内公式}
大家好，我是$a^2+b^=c^2$行内公式。
```

![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")

显示结果：

![](https://img-blog.csdnimg.cn/20210705084444496.png)![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")​编辑

 **常用的数学符号的LaTex表示：**

字符/运算符

LaTex表示形式

![\alpha](https://latex.codecogs.com/gif.latex?%5Calpha)![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")​编辑

\alpha

![\beta](https://latex.codecogs.com/gif.latex?%5Cbeta)![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")​编辑

\beta

![\gamma](https://latex.codecogs.com/gif.latex?%5Cgamma)![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")​编辑

\gamma

![\delta](https://latex.codecogs.com/gif.latex?%5Cdelta)![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")​编辑

\delta

![\theta](https://latex.codecogs.com/gif.latex?%5Ctheta)![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")​编辑

\theta

![\lambda](https://latex.codecogs.com/gif.latex?%5Clambda)![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")​编辑

\lambda

![\sigma](https://latex.codecogs.com/gif.latex?%5Csigma)![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")​编辑

\sigma

![\sum](https://latex.codecogs.com/gif.latex?%5Csum)![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")​编辑(求和符号)

\sum

![\prod](https://latex.codecogs.com/gif.latex?%5Cprod)![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")​编辑

\prod

![\lim](https://latex.codecogs.com/gif.latex?%5Clim)![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")​编辑(极限符号)

\lim

![\left | a \right|](https://latex.codecogs.com/gif.latex?%5Cleft%20%7C%20a%20%5Cright%7C)![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")​编辑(a的绝对值)

\left | a \right|

![\overline{A}](https://latex.codecogs.com/gif.latex?%5Coverline%7BA%7D)![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")​编辑

\overline{A}

![\widetilde{y}](https://latex.codecogs.com/gif.latex?%5Cwidetilde%7By%7D)![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")​编辑

\widetilde{y}

![a^2](https://latex.codecogs.com/gif.latex?a%5E2)![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")​编辑（上标的表示）

^{}

![A_i](https://latex.codecogs.com/gif.latex?A_i)![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")​编辑(下标的表示)

A_i

![\frac{1}{2}](https://latex.codecogs.com/gif.latex?%5Cfrac%7B1%7D%7B2%7D)![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")​编辑（分式）

\frac{1}{2}

对以上表格做一些简单解释：

单个的希腊符号显示比较简单，直接看求和符号：

```
1 * 1 + 2 * 2 + ...... + 100 * 100 = $\sum _i=1 ^n=100 x_i ^2$
```

![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")

运行结果：

![](https://img-blog.csdnimg.cn/20210705091123437.png)![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")​编辑

此时，我们看到显示结果和预期结果不一致，这是因为使用上下标时，如果上下标后面的字符过多，我们就需要加 '{}' 括住。修改后如下：

```
1 * 1 + 2 * 2 + ...... + 100 * 100 = $\sum _{i=1} ^{n=100} x_i ^2$
```

![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")

显示结果：

![](https://img-blog.csdnimg.cn/20210705091349233.png)![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")​编辑

这是在行间公式中的显示，我们能否把下标 ![i=1](https://latex.codecogs.com/gif.latex?i%3D1)![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")​编辑 显示到求和符号的最下面，行间公式中，只需要在求和符号下面增加 \lims 即可，测试如下：

```
1 * 1 + 2 * 2 + ...... + 100 * 100 = $\sum \limits _{i=1} ^{n} x_i ^2 (n = 100)$ 
```

![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")

![](https://img-blog.csdnimg.cn/20210705091725110.png)![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")​编辑

表格中的其他符号可以以此类推。

### 1.3.2 行间公式

语法格式：

```
\[
    内容
\]
```

![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")

测试：输出勾股定理的公式：

```
\subsection{行间公式}
勾股定理的公式如下所示：
\[
	a ^ 2 + b^2 = c^2
\]
这是下一行！
```

![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")

显示如下：

![](https://img-blog.csdnimg.cn/20210705092053713.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3lfaF9rXzY2Ng==,size_16,color_FFFFFF,t_70)![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")​编辑

行间公式也就意味着：公式单独占一行。

行间公式中使用求和符号：

```
我马上要使用行间公式了，是不是有点激动？
\[
	MAE = \frac{1}{n} \sum _{i=1} ^{n} \left| \widetilde{y} _{i} - y _{i} \right|
\]
结束了，来看看效果！
```

![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")

![](https://img-blog.csdnimg.cn/20210705092826493.png)![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")​编辑

注意：行间公式中，不用使用 \limits 下标 ![i=1](https://latex.codecogs.com/gif.latex?i%3D1)![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")​编辑 就在最下面。以上代码应该都可以看懂。

### 1.3.3 行间公式自动 加编号

语法：使用 \begin{equation} \end{equation}，中间写公式代码

```
\subsection{行间公式自动编号}
\begin{equation}     
	a ^ 2 + b^2 = c^2
\end{equation}
```

![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")

![](https://img-blog.csdnimg.cn/20210705093411448.png)![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")​编辑

对于写论文时，这个功能是相当方便的。

继续添加公式再试一下：

```
\begin{equation}
	1+2+3+\dots+(n-1)+n = \frac{n(n+1)}{2}
\end{equation}
```

![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")

![](https://img-blog.csdnimg.cn/20210705093608150.png)![](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw== "点击并拖拽以移动")​编辑

标号会自动增加。

        编译器我是用的是 texworks，大家可以到官网下载，也可以使用typora的插入公式功能去测试。当然也可以用VScode添加插件编写LaTex代码，下篇我会写一下如何配置 VScode 编写LaTex 代码。本篇用来记录一下知识点，方便本人忘记时查看，也供大家参考，有问题希望朋友门指正，感觉写的对你有帮助别忘记点个关注哈。

  

​