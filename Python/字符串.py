
def match_KMP(t, p, gen_pnext):
    """:arg
       t : 目标串
       p : 模式串
       next : 模式串的next数组
    """
    i, j = 0, 0
    n, m = len(t), len(p)   # m 、 n 分别是模式串 和 目标串的长度
    while i < m and j < n:
        if i == -1 or t[j] == p[i]:
            i, j = i+1, j+1
        else:
            i = gen_pnext[i]
    if i == m:              # 找到匹配的子字符串，返回其下标
         return j-i
    return -1

def gen_pnext(p):
    i, k, m = 0, -1, len(p)
    pnext = [-1] * m
    while i < m-1:
        if p[i] == p[k] or k == -1:
            i, k = i+1, k+1
            pnext[i] = k
        else:
            k = pnext[k]
    return pnext
""":arg
1.找到被匹配字符串的次数，计算新字符串需要的长度
2.把字符串添加到新字符串中
"""
def replace(t, pattern, string):
    ''':arg
        t : 目标串
        pattern ： 模式串
        string : 被替换串
    '''
    count = 0
    j = 0
    while j < len(t):
        j = match_KMP(t, pattern, gen_pnext(pattern))
        count, j = count+1, j+len(pattern)
        t = t[j:]
    print("t:", t)
    if t == t[-1]:
        count = 0
    print("count:", count)
    pass

if __name__ == '__main__':
    t, p = 'I love China ov', 'ov'
    res = match_KMP(t, p, gen_pnext(p))
    print(res)
    # replace(t, p, 'BB')
    res = t.replace('ov', 'AA')
    print(res)
