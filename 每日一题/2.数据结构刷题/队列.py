""":arg
    队列（queue）或者称为队，是一种容器可以存入、访问、删除元素。
    特点：
        先进先出（First In First Out， 简写为 FIFO）
    将一个元素放入队列称为 入队（enqueue），从队列中删除一个元素并返回它称为出队（dequeue）.

    队列的实现：
    ADT Queue:
        Queue(self)         # 创建空队列
        is_empty(self)      # 判空
        enqueue(self, elem)       # 将元素 elem 加入队列， 称为入队
        dequeue(self)             # 删除队列中最早进入的元素并将其返回，称为出队
        peek(self)                # 查看队列里最早进入的元素，不删除
"""

class QueueUnderFlow(ValueError):
    ''':arg
        队列为空时，则无法进行操作，抛出异常
    '''
    pass

# 队列类的实现
class SQueue():
    ''':arg
        类内部 使用 列表存储，如果入队超过 len， 则把列表的长度扩大，把原来的元素赋值到新列表里

        在频繁出队、入队之后，有可能存储在队尾的元素已经满了，但是队首的元素还未存储满，因此，我们可以使用循环列表进行优化，
        以便使用队首的空白存储区
    '''
    def __init__(self, init_len=8):
        self._len = init_len        # [] 初始化的长度
        self._elems = [0]*init_len  # [] 里的元素初试化为 0
        self._head = 0              # [] 设第一个节点的下标为 _head
        self._num = 0               # 队列中元素个数

    def is_empty(self):
        return self._num == 0

    def peek(self):
        if self._num == 0:
            raise QueueUnderFlow
        return self._elems[self._head]

    def dequeue(self):
        if self._num == 0:
            raise QueueUnderFlow
        e = self._elems[self._head]                 # 保存出队元素，一会返回
        self._head = (self._head+1) % self._len
        self._num -= 1                              # 出队一次， 元素少1
        return e

    def enqueue(self, e):
        if self._num == self._len:
            self.__extend()                         # 此时列表已经满，需要扩充元素
        self._elems[(self._head + self._num) % self._len] = e
        self._num += 1
        pass

    # 该方法将存储区长度加倍
    def __extend(self):
        old_len = self._len
        self._len *= 2              # 新列表长度扩大2倍
        new_elems = [0]*self._len
        # 旧列表元素复制到新列表
        for i in range(old_len):
            new_elems[i] = self._elems[(self._head+i)%old_len]
        self._elems, self._head = new_elems, 0
        pass
    pass

if __name__ == '__main__':
    queue = SQueue()
    queue.enqueue(10)
    print(queue.peek())
    queue.enqueue(20)
    print(queue.peek())
    queue.dequeue()
    print(queue.peek())

