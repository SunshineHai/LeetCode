import math

# 1. 有理数类
# 封装一个类，实现2个分数相加
class Rational0:

    """:arg
        该类的缺点：结果没有进行约分
    """
    # 构造方法，实例化对象时调用
    def __init__(self, num, den=1):
        self.num = num      # 分子    公共的, 实例对象可以访问
        self.den = den      # 分母

    # 实例方法：实现两个分数相加
    def plus(self, another):
        den = self.den * another.den                            # 新分母
        num = self.num * another.den + self.den * another.num   # 新分子
        return Rational0(num, den)      # 返回相加后的 实例对象

    def print(self):
        print(str(self.num) + '/' + str(self.den))

# 2.优化
class  Rational1:
    """:arg
        1.考虑输入时分母不为0，输入数字有正负之分，我们统一规定输入的分母都为正，分子为正则为正分数，分子为负则为负分数
        2.对结果进行约分时，我们需要找到醉倒公约数，该方法的定义是局部使用的，可以说是可以分离出来的，我们定义静态方法
            方法前加入： @staticmethod
            方法里去掉   self
            静态方法的调用： 类名.静态方法()  或者 实例对象.静态方法()
        本质上说 静态方法是类里面定义的普通函数，但也是该类的局部函数。

        辗转相除法求2个数的最大公约数：
        例：求 80 和 75 的最大公约数：
        80 / 75 = 1......5
        75 / 5 = 15......0
        5 / 0
        则最大公约数为 5

        30 % 5 = 0
        5 % 0
    """
    @staticmethod
    def _gcd(m, n):     # _ : 方法名前单个下划线：当做内部使用的名字，不能在这个类外使用
        if m < n:
            m, n = n, m     # 保证 m > n
        while n != 0:
            m, n = n, m % n
        return m            # 当 余数为0时， m 的值即为这m/n的最大公约数

    def __init__(self, num, den=1):
        if not isinstance(num, int) or not isinstance(den, int):
            raise TypeError
        if den == 0:
            raise ZeroDivisionError
        sign = 1        # 分母正负的标志
        if num < 0:
            num, sign = -num, -sign
        if den < 0:
            den, sign = -den, -sign
        g = Rational1._gcd(num, den)
        # print(f'g:{g}')

        self._num = sign * (num // g)
        self._den = den // g

    def print(self):
        print(str(self._num) + '/' + str(self._den))

#   成员变量是私有的，只能在本类中使用，不能 对象.成员变量名 调用，如果要使用成员变量可以定义实例方法
    def num(self):
        return self._num

    def den(self):
        return self._den

#   有理数的运算，Python中的特殊方法名：都以2个下划线开始，并以2个下划线结束。
#     +
    def __add__(self, another):
        den = self._den * another.den()     # 通分
        num = (self._num * another.den() + self._den * another.num())
        return Rational1(num, den)

#     -
    def __sub__(self, another):
        num = self.num * another.den() - self._den * another.num()
        den = self._den * another.den()
        return Rational1(num, den)

    # *
    def __mul__(self, another):
        return Rational1(self._num * another.num(), self._den * another.den())

    # 整除 ： //
    def __floordiv__(self, another):
        if another.num() == 0:
            raise ZeroDivisionError
        return Rational1(self._num * another.den(), self._den * another.den())

#   比较有理数对象相等或不等
#     == : __eq__
    def __eq__(self, another):
        return self._num * another.den() == self._den * another.num()

#     < : __lt__
    def __lt__(self, another):
        return self._num * another.den() < self._den * another.num()

#     > : __gt__
    def __gt__(self, another):
        return self._num * another.den() > self._den * another.num()

#     != : __ne__
    def __ne__(self, another):
        return self._num * another.den() != self._den * another.num()

#     <= : __le__
    def __le__(self, another):
        return self._num * another.den() <= self._den * another.num()

#     >= : __ge__
    def __ge__(self, another):
        return self._num * another.den() >= self._den * another.num()

#   定义把类的对象 转化成 字符串： __str__ , 内置 str 将会调用它
    def __str__(self):
        return str(self._num) + "/" + str(self._den)

    def print(self):
        print(type(self._num), type(self._den))
        print(self._num, "/", self._den)
    pass

''' 
    类的定义和使用：
        1.类定义
            class <类名>
                <语句组>
        2.类对象及其使用
            执行一个类定义将创建一个类对象
                1.属性访问(成员变量)
                2.实例化
        3.实例对象 ： 初始化和使用
            创建一个类的实例对象：实例对象名 = 类名(属性)，将自动调用 类中的 __init__() 方法完成初始化
'''

class C:
    """:arg
        说明：
            Python 中提供了一个内置函数 isinstance(),专门用来检查类和对象的关系。
            isinstance(obj, cls) : 检测对象obj是否为类cls的实例，是则返回True，否则返回False；
            也可以检测 任何对象 和 任何类型的关系 eg:
            isinstance(12, int) : 判断一个变量是否为int型
    """
    def __init__(self, m=1, n=1):
        self._m = m       # _m 表示 该变量是私有的，只能在类中使用
        self._n = n       # 成员变量，又叫 数据属性
        self.__x = 10
    # 定义方法函数，self 表示 本实例对象，self 是约定俗称的写法，可以写其他的my等
    def m(self, a, b):
        return self._m * a + self._n * b
    pass

class Countable:
    """:arg
        1.静态方法
            1.定义：def 行前 加修饰符 @staticmethod
            2.静态方法是 普通函数，只是由于某种原因需要定义在类里面。
            3.静态方法的参数可以根据需要定义，不需要特殊的 self 参数
            4.调用：通过 类名.静态方法名() 或者 实例对象.静态方法名()
        eg ： Rational1 类 中的 _gcd() 方法即是静态方法。

        2. 定义一个计数器，返回 该类中实例化多少个对象

        get_count() 即是一个 类方法：
            1.定义形式为 def 前， 加修饰符 @classmethod
            2.这种方法必须有一个表示其调用类的参数，习惯用 cls 作为参数名，还可以有任意多个其它属性
            3.类方法 也是 类对象的 属性， 可以 以 属性访问形式调用
            4.在类方法执行时，调用它的类将自动约束到方方的cls参数，可以通过这个参数访问该类的其他属性
    """
    counter = 0

    def __init__(self):
        Countable.counter += 1

    @classmethod
    def get_count(cls):
        return Countable.counter
    pass

if __name__ == '__main__':
    r1 = Rational0(3, 5)
    r2 = r1.plus(Rational0(7, 15))
    r2.print()
    print(r2.num)

    print('res:', Rational1._gcd(75, 80))

    r3 = Rational1(5, 30)
    r3.print()

    print("---------------------------")
    five = Rational1(5)   # 初始化是只有分子，用整数创建有理数，分母默认为 1
    five.print()
    x = Rational1(3, 5)
    x.print()

    # 由于 有理数类 定义了 str 转化函数， 可以直接使用print输出：此时调用的就是 __str__() 函数，把成员变量转换成字符类型输出
    print("Two thirds are", Rational1(2, 3))

#     对于有理数，可以使用类中定义的算术运算符和条件运算符
    y = five + x * Rational1(5, 17)         # 对象相乘， 直接调用 __mul__() 方法
    if y < Rational1(123, 11):
        print("y is small! ")
    else:
        print("y is bigger")
    t = type(five)
    print(t)
    if isinstance(five, Rational1):
        print("Yes")
    else:
        print("No")

    print(math.sin(math.pi/2))

    print(Rational1.__doc__)    # 类中的注释

#     类的定义和使用
    o = C(2, 3)     # o 是 类C的实例对象

#     调用 类C中的方法函数 m
    o.m(1, 2)
    print(C.m)          # C.m 是一个普通的函数对象
    p = C(3, 4)         # p 是类C的一个实例
    print(p.m)          # p.m 是基于 这个实例 和 函数 m 建立的一个 方法对象

    """ 使用方法对象的方法：
            1.类实例做方法调用
                eg: p.m(a, b) 即可
            2. p.m(a, b) 等价于 C.m(p, a, b)
                方法中的其他参数可以通过调用表达式中其他参数提供
            3.方法对象也是一种类似函数的对象，可以作为对象使用
                eg: 可以把方法对象赋值给变量，或者作为实参传入函数，然后在函数的其他地方作为函数去调用
                q = p.m
                q(a, b)
                表示 a, b 作为实参调用 p.m() 这个方法   
            以上三种方法是等价的                 
    """
    print(p.m(2, 3))
    print(C.m(p, 2, 3))
    q = p.m
    print(q(2, 3))

#     Countable 类 的实例化
    x = Countable()
    y = Countable()
    z = Countable()

    print(Countable.get_count())

    x1 = C(1, 2)
    # print(C.__x)  # __m 的变量， Python解释器会做统一的改名，类之外采用属性访问无法找到它





