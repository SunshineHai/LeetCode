import pandas as pd
import numpy as np


def normalization(data):
    minVals = data.min(axis=0)  # 计算每一列的最小值
    maxVals = data.max(axis=0)  # 计算每一列的最大值
    ranges = maxVals - minVals  # 最大值-最小值
    m = data.shape[0]           # m 表示 行
    normData = data - np.tile(minVals, (m, 1))      # 把 minVals 转化为 [1, 2, 3] 扩展为 m行1份 值和第一行一样
    normData = normData / np.tile(ranges, (m, 1))   # (X-X_min) / (max-min)
    return normData, ranges, minVals


data1 = pd.read_excel(r'data/abnormal.xlsx')
data2 = pd.read_excel(r'data/normal.xlsx')
df = pd.concat([data1, data2]).reset_index()  # 按照行合并，并重置索引
print(df.shape)
print(df.columns)
# print(df)
df = df.iloc[:, 1:]
# 数据预处理

# 1.对于全部数据 删除缺失值比例大于50%的特征例
data = df
missing_cols = [c for c in data if data[c].isna().mean()*100 > 50]
data = data.drop(missing_cols, axis=1)

# 2.对object型的缺失值进行填充   :  字符类型的
object_df = data.select_dtypes(include=['object'])     # 字符型的列数据
numerical_df = data.select_dtypes(exclude=['object'])  # 数值型的列数据

object_df = object_df.fillna('unknow') # 把列中的 object 对象填充为字符串 "unknow"

# 3.每列为空的数的个数 > 0 的列 ： 列中含有空值的所有列
missing_cols = [c for c in numerical_df if numerical_df[c].isna().sum() > 0]
# 4.对于数值型，含空的数值填充为该列的中位数
for c in missing_cols:
    numerical_df[c] = numerical_df[c].fillna(numerical_df[c].median())

if object_df.empty == False:
    print('执行中......')
    data = pd.concat([object_df, numerical_df], axis=0).reset_index()
else:
    data = numerical_df

print("-----------------------------------")

data = data.drop('序号', axis=1)
print(data)
data.to_csv(r'data\distance.csv')
