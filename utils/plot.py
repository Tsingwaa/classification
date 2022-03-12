from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import pandas as pd
# x = []
# y = []
# x1 = 1.15
# x2 = 0.9

# for i in range(1000):
#     x.append(i)
#     if i> 860:
#         y.append(-0.00000398*(i-860)*(i-860)+0.978+np.random.randn()*0.013)
#     else:
#         y.append(-0.0002*i+1.15+np.random.randn()*0.01)
# plt.ylim(0.7, 1.3)
# plt.plot(x, y)
# plt.show()
# plt.savefig('a.jpg')

# src = cv.imread('C:\Users\lu\Desktop\lena.jpg')
# src = cv.imread('C:\\Users\\lu\\Desktop\\lena.jpg')
# src = cv.cvtColor(src, cv.COLOR_BGR2RGB)

# dst = src[]
# cv.imwrite('C:\\Users\\lu\\Desktop\\lena_h.jpg', cv.cvtColor(dst, cv.COLOR_RGB2BGR))

# dst = cv.flip(src, 0)
# cv.imwrite('C:\\Users\\lu\\Desktop\\lena_v.jpg', cv.cvtColor(dst, cv.COLOR_RGB2BGR))

from matplotlib.pyplot import MultipleLocator
x_major_locator=MultipleLocator(0.01)
y_major_locator=MultipleLocator(5)

plt.figure(figsize=(4,3))
# plt.rc('font',family='Times New Roman')
data = pd.read_csv("C:\\Users\\lu\\Desktop\\test.csv")
x = list(data['temp'].values)
y = list(data['mr'].values)

y2 = [84.11] * len(x)
y3 = [85.21] * len(x)


ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)

plt.xlim(0, 0.1)
plt.ylim(70, 100)
plt.xlabel("Temperature")
plt.ylabel("Mean Class Accuracy")

# plt.grid(True)
# plt.grid(color='black',    
#          linestyle='--',
#          linewidth=1,
#          alpha=0.1) 
x2 = [0.05] * 5
yy = [60, 70, 80, 90, 100]
plt.plot(x, y, color="#2E8B57", marker='o', ms=1, lw=1)
plt.plot(x, y2, '--', color="#EEB422", lw=1)
plt.plot(x, y3, '-.', color="#1E90FF", lw=1)
plt.plot([0.05], [86.43],  marker='^', color="r", lw=1)

plt.show()
