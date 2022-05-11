import requests
from bs4 import BeautifulSoup
import datetime

'''1.获取系统日期'''
today = datetime.date.today()
print(today)

print('---------------开始-----------------------')
url = 'http://sjxy.whpu.edu.cn/index/tzgg.htm'
# 模拟浏览器发送HTTP请求
header = {'User-Agent': 'Mozilla/5.0'}
try:
    response = requests.get(url, headers=header)
    response.raise_for_status()
    # 设置编码
    response.encoding = response.apparent_encoding
    # 获取 网页 中的 HTML 源码
    html = response.text
    soup = BeautifulSoup(html, "html.parser")
    # print('soup:',soup)
    target = soup.find_all("p", string = '2020/06/01')
    # target为列表
    for eachOne in target:
        each_text = eachOne.parent.parent.td.a.text
        each_href = eachOne.parent.parent.td.a.get("href")
        print(each_text)
        print('http://sjxy.whpu.edu.cn' + each_href)
except:
    print("爬取失败")

print('---------------结束-----------------------')

