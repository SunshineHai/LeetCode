import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns





data = pd.read_csv(r'..\data\distance.csv')
print(data.head())

corrmat = data.corr()
f, ax = plt.subplots(figsize = (20, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)
plt.savefig(r"..\figure\相似性矩阵.png")