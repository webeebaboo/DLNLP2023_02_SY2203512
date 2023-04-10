import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# # 定义高斯分布的参数
# mean1, std1 = 164, 3
# mean2, std2 = 176, 5
#
# # 从两个高斯分布中生成各50个样本
# data1 = np.random.normal(mean1, std1, 500)
# data2 = np.random.normal(mean2, std2, 1500)
# data = np.concatenate((data1, data2), axis=0)
#
# # 将数据写入 CSV 文件
# df = pd.DataFrame(data, columns=['height'])
# df.to_csv('height_data.csv', index=False)

# 绘制数据的直方图
# plt.hist(data, bins=20)
# plt.xlabel('Height (cm)')
# plt.ylabel('Count')
# plt.title('Distribution of Heights')
# plt.show()


# 读取数据
fr= pd.read_csv('height_data.csv')
height = fr['height'].values


# 定义高斯函数
def gaussian(x, mu, sigma):
    func=1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-0.5 * (x - mu) ** 2 / sigma ** 2)
    return func


# 给定初始参数
mu1, sigma1 = 170, 5
mu2, sigma2 = 180, 5
pi = 0.5

# 迭代次数
j = 1000

# EM算法
for i in range(j):
    # E
    gamma1 = pi * gaussian(height, mu1, sigma1)
    gamma2 = (1 - pi) * gaussian(height, mu2, sigma2)
    gamma_sum = gamma1 + gamma2
    gamma1 /= gamma_sum
    gamma2 /= gamma_sum

    # M
    mu1 = np.sum(gamma1 * height) / np.sum(gamma1)
    mu2 = np.sum(gamma2 * height) / np.sum(gamma2)
    sigma1 = np.sqrt(np.sum(gamma1 * (height - mu1) ** 2) / np.sum(gamma1))
    sigma2 = np.sqrt(np.sum(gamma2 * (height - mu2) ** 2) / np.sum(gamma2))
    pi = np.mean(gamma1)

    lenn0 = np.sum(gamma1)
    lenn1 = len(height) - lenn0
    mu1 = gamma1.dot(height) / lenn0
    mu2 = gamma1.dot(height) / lenn1
    sigma1 = np.sqrt(gamma1.dot((height - mu1) ** 2) / lenn0)
    sigma2 = np.sqrt(gamma1.dot((height - mu2) ** 2) / lenn1)
    pi = lenn0 / len(height)




x = np.linspace(150, 195, 5000)
y = pi * gaussian(x, mu1, sigma1) + (1-pi) * gaussian(x, mu2, sigma2)
plt.hist(height, bins=50, density=True, alpha=0.5)
plt.plot(x, y, 'g-', linewidth=2)
plt.show()