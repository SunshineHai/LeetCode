#  egmentation fault C语言
## 看如下程序：
```c
#include<stdio.h>
#include<string.h>
void swap(char *x, char *y){
    char temp = *x; 
    *x = *y; 
    *y = temp;
}
 
 
void main(){
    char *str = "apple";  // 这里当修改字符内容时3会有问题
    swap(str, str+4);
} 
```
![](https://s2.loli.net/2022/03/27/1QeGnSXW3oi5bhf.png)
昨天遇到了如上的问题，经过看别人的回答和查阅书籍才明白，如下：

**注意：** 对于 `` char *str = "apple" ``  "apple" 存储在 **静态存储器(static memory)**是不可以被修改的，指针指向了该字符串首字符的地址，静态存储区的内容是*不允许修改的*。对于使用数组初始化：`` char str[] = "apple" `` ，在程序运行时把静态存储区的字符串赋值给数组，而数组元素是可以修改的。

-  当时我把主要精力放在调试temp函数里，认为是temp函数出现问题，实际上则不然。
-  赋值的时候出现了问题，我就应该考虑是不是不可以赋值去修改字符串里的内容
-  这时候在考虑 字符串 存放到了哪里，可以修改吗?

因此这个程序只需要把 `` char *str = "apple"`` 修改为 `` char str[] = "apple" `` 即可。如下：

```c
#include<stdio.h>
#include<string.h>
void swap(char *x, char *y){
    char temp = *x; 
    *x = *y; 
    *y = temp;
}
 
 
void main(){
//    char *str = "apple";  // 这里当修改字符内容时3会有问题
	char str[] = "apple";
    swap(str, str+4);
    puts(str);
} 
```

![](https://s2.loli.net/2022/03/27/Ev7nPWIe3yDwA8r.png)

对于昨天的刷题文章里出现的问题就可以解决了，程序如下：
字符串反转：
```c
#include<stdio.h>
#include<string.h>
void swap(char *x, char *y){
	char temp = *x; 
	*x = *y; 
	*y = temp;
}

void reverseString(char	*s, int sSize){
	int left = 0;
	int right = sSize-1;
	while(left < right){
		swap(s+left, s+right);
		left++;
		right--;
	}
}

void main(){
	char str[] = "apple";  //这个语句修改一下就可以了 
	int len = strlen(str);
	reverseString(str, len);
	printf("%s\n", str);
} 


```
![](https://s2.loli.net/2022/03/27/sEMnq8Gp4kz9FgD.png)
今天又解决了一个问题，记录一下，哈哈哈。` printf() `

