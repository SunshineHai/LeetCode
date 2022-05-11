from sklearn.svm import SVC             # 支持向量机回归
from sklearn.metrics import roc_curve
from sklearn.datasets import make_blobs
from sklearn. model_selection import train_test_split
import matplotlib.pyplot as plt

# 生成一个二分类的数据不平衡数据集
X, y = make_blobs(n_samples=(4000, 500), cluster_std=[7, 2], random_state=0)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# 支持向量机模型使用的decision_function函数，是自己所特有的，而其他模型不能直接使用。
clf = SVC(gamma=0.05).fit(X_train, y_train)

fpr, tpr, thresholds = roc_curve(y_test, clf.decision_function(X_test))

plt.plot(fpr, tpr, label='ROC')

plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title("ROC curve")
plt.savefig("..\\\\figure\\ROC.png", dpi=300)
plt.show()