import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

path = '.\ex7faces'
data = loadmat(path)
data = data['X']
'''
print(data,type(data),data.shape)
'''

# 1. 均值归一化
X_center = data - np.mean(data,axis=0)  # 按行操作，取每一列的均值

# 2.求协方差矩阵
C = np.dot(np.transpose(X_center), X_center) / len(X_center)
'''
print('协方差矩阵：',C)
'''

# 3.求协方差矩阵的特征值和特征向量

# 特征值分解
a = np.linalg.eig(C)
print('特征值为：',a[0],'特征向量为：',a[1])
'''
# 奇异值分解
U, S, V = np.linalg.svd(C)
print("特征值：", S)
print("特征向量：", U)
'''
print('ooookkkk')
# 4.降维，特征值从大到小排序
D = a[0]
V = a[1]
sorted_indices = np.argsort(D)
V1 = V[:, sorted_indices[:-100 - 1:-1]]
print(V1,V1.shape)
print(X_center.shape)
X_reduction = np.dot(X_center, V1)
print("降维后的数据形状：", X_reduction.shape)

# 5. 数据还原
# x做过均值归一化，因此还需要加上各个维度的均值。
print(X_reduction.shape)
print(V1.T.shape)
X_restore = np.dot(X_reduction, V1.T) + np.mean(data, axis=0)

# 画图
# 原图
fig, axis = plt.subplots(ncols=10, nrows=10, figsize=(10, 10))
for c in range(10):
    for r in range(10):
        # 显示单通道的灰度图像, cmap='Greys_r'
        axis[c, r].imshow(data[10*c + r].reshape(32, 32).T)
        axis[c, r].set_xticks([])
        axis[c, r].set_yticks([])
plt.show()
# 处理后的图
fig, axis = plt.subplots(ncols=10, nrows=10, figsize=(10, 10))
for c in range(10):
    for r in range(10):
        # 显示单通道的灰度图像, cmap='Greys_r'
        axis[c, r].imshow(X_restore[10*c + r].reshape(32, 32).T)
        axis[c, r].set_xticks([])
        axis[c, r].set_yticks([])
plt.show()

