import time
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

currdate = dt.date.today()
currdate = str(currdate).replace('-', '/')

print(currdate)

df = pd.DataFrame(np.array([
    ['Alfred', '0', np.NaN],
    ['Batman', pd.NaT, '7'],
    [np.nan, '10', '11'],
    ['I', "love", "China"]
]), columns=['name', 'toy', 'born'])
print(df)

print('--------------------------------')
print(df.dropna(axis=1))

from sklearn.preprocessing import StandardScaler

X_train = pd.DataFrame({'a' : [3, 2, 7],
                        'b' : [6, 4, 9]})
print(X_train)
transfer = StandardScaler()                    #  实例化对象
X_train = transfer.fit_transform(X_train)

print(X_train)

name = "China"
save_path = "..\\\\figure\\" + name + ".png"
print(save_path)

x, y = 3, 5
x_list = [i for i in range(x)]
print(x_list)

y_list = [i for i in range(x, x+y)]
print(y_list)


def show_error(row_name:list, precision:list, recall:list, f1:list, AUC:list, path):
#     plt.figure(figsize=(10, 10))
    plt.plot(row_name, precision, 'or-', recall, '^g-', f1, '*b-', AUC, 'bo')
    plt.legend(['precision', 'recall', 'f1', 'AUC'], loc='upper right')
    plt.savefig(path, dpi=300)
    plt.show()
pass

x_name = ['precision', 'recall', 'f1', 'AUC']

show_error(x_name, [0.5, 0.6, 0.8, 0.9], [0.56, 0.66, 0.82, 0.91], [0.4, 0.6, 0.85, 0.92], [0.6, 0.7, 0.82, 0.84, 0.93], r'test.png')