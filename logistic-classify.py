#对数几率回归是用来分类的，根据多个参数对结果进行分类
#y=1/(1+exp-(w1*x1+w2*x2+b))其中x是输入参数，y是输出参数，当y<0.5判为0，当y>0.5判为1


import numpy as np
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

data = np.array([[0.697, 0.460, 1],
                 [0.774, 0.376, 1],
                 [0.634, 0.264, 1],
                 [0.608, 0.318, 1],
                 [0.556, 0.215, 1],
                 [0.403, 0.237, 1],
                 [0.481, 0.149, 1],
                 [0.437, 0.211, 1],
                 [0.666, 0.091, 0],
                 [0.243, 0.267, 0],
                 [0.245, 0.057, 0],
                 [0.343, 0.099, 0],
                 [0.639, 0.161, 0],
                 [0.657, 0.198, 0],
                 [0.360, 0.370, 0],
                 [0.593, 0.042, 0],
                 [0.719, 0.103, 0]])
X = data[:, 0:2]
Y = data[:, -1]

# 绘制数据集
f1 = plt.figure(1)
plt.title("watermelon_3a")
plt.xlabel("密度")
plt.ylabel("含糖量")
plt.scatter(X[Y == 0, 0], X[Y == 0, 1], marker='o', color='k', s=100, label='bad')
plt.scatter(X[Y == 1, 0], X[Y == 1, 1], marker='o', color='g', s=100, label='good')
plt.legend(loc='upper right')
plt.show()

# 使用sklearn

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.5, random_state=0)

log_model = LogisticRegression()
log_model.fit(X_train, Y_train)

Y_pred = log_model.predict(X_test)

print(metrics.confusion_matrix(Y_test, Y_pred))
#这是打印出真假矩阵
print(metrics.classification_report(Y_test, Y_pred))
#这是打印出分类报告
#              precision    recall  f1-score   support
#         0.0       0.75      0.60      0.67         5  （5个被判为真，其中只有3个是真的真，60%）
#        1.0       0.60      0.75      0.67         4    （4个被盘问假的，有3个是真的假，75%）
#   accuracy                           0.67         9
#   macro avg       0.68      0.68      0.67         9
#weighted avg       0.68      0.67      0.67         9

print(log_model.coef_)#求解参数w
print(log_model.intercept_) #偏置b
#y=1/(1+exp-(0.075*x1+0.398*x2+b))这个是最终推出来的式子

print(Y_pred)
right = 0
for i in range(len(Y_pred)):
    if Y_pred[i] == Y_test[i]:
        right += 1
print("accuect is :", right /(len(Y_pred)))
#显示成功率为67%