import re



""":arg
    \d : 等价于 [0-9]
    \. : 匹配 . 
"""
def p_words(string):
    string_list = re.findall(r"\d+\.\d+", string)  # 找到 string 里与 pattern 匹配的字串, 批量 数字.数字
    return string_list[0]


if __name__ == '__main__':
    res = p_words("I love China! 1.2, 3.4, 我爱你,3.1415926/")
    print(res)