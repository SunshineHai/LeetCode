
""":arg
栈：是一种常用的数据结构，特点是：先入后出
实现方法：
    1.顺序表
    2.链表
"""

# 顺序表
""":arg
    顺序表的实现：
    列表本身可以看做是用栈实现的，但是 list.pop(index=),形参的索引为空时可以看做出栈，但是list提供可以删除指定索引的元素，
    因此我们使用列表定义严格意义的栈。
        1.把列表 [] 作为私有的成员变量
        2.栈只能操作栈顶元素，我们根据顺序表的特点，把顺序表的尾部作为栈顶，操作式复杂度为O(1) 
    定义如下：
"""
# 定义异常：栈下溢
class StackUnderflow(ValueError):
    pass

class Sstack:

    def __init__(self):
        self._elems = []

    def is_empty(self):
        return self._elems == []        # 加下划线，代表该成员变量为私有变量

    # 返回栈顶元素
    def top(self):
        if self.is_empty():
            raise StackUnderflow("in Sstack.top()")   # 使用 raise 抛出自定义异常
        else:
            return self._elems[-1]    # 返回栈顶，即最后一个元素

    def push(self, element):
        self._elems.append(element)

    def pop(self):
        if self.is_empty():
            raise StackUnderflow("in SStack.pop()")
        else:
            return self._elems.pop()    # 列表中的pop()，移除列表中最后一个元素，并返回移除的元素

""":arg
    由链表实现栈, 根据链表的特性，我们应该把第一个结点作为栈顶元素而不是尾结点
    1.栈顶指向头结点
    2.实现常用操作
    
"""
class Node:
    def __init__(self, data, next_):
        self.data = data
        self.next_ = next_
    pass


class LStack:
    def __init__(self):
        self._top = None       # 初始化时，头结点为空, 私有成员变量

    def is_empty(self):
        return self._top is None

    # 返回栈顶元素
    def top(self):
        if self._top is None:
            raise StackUnderflow("error in Lstack.top()")
        else:
            return self._top.data

    def push(self, element):
        self._top = Node(element, self._top)  # 新节点指向了旧结点


    def pop(self):
        if self.is_empty():
            raise StackUnderflow("error in pop()")
        else:
            e = self._top.data
            self._top = self._top.next_    # 空出来的结点系统会自动释放
            return e

if __name__ == '__main__':
    s = Sstack()
    print(s.is_empty())
    s.push(1)
    s.push(2)
    s.push(3)
    print(s)
    print(s.top())
    s.pop()
    print(s.top())

    print('-------------------------')
#   遍历输出：
    while not s.is_empty():
        print(s.pop())

    print(s.is_empty())
    # print(s.pop())
    s.push(10)
    print(s._elems)

    my_list = [1]
    print(my_list.pop())

    ss = LStack()
    print(ss.is_empty())
    ss.push(10)
    print('res:', ss.top())
    ss.push(20)
    ss.push(30)


    # 变量输出
    while not ss.is_empty():
        print(ss.pop(), end='\t')
    print()
