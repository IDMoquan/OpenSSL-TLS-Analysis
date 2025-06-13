import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "SimHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

df = pd.read_csv("TLS\Tain_Accuracy_300.csv", header=None)
value = np.array(df[1])
loss = np.array(df[2])
x = np.arange(1,301,1)
plt.figure(figsize=(8,5))
plt.plot(x, value, "r-", label="正确率")
plt.plot
plt.grid()
plt.legend()
plt.title("预测正确率")
plt.xlabel("训练轮次/epochs")
plt.ylabel("正确率/%")
plt.show()
plt.figure(figsize=(8,5))
plt.plot(x, loss, "b-", label="平均损失")
plt.plot
plt.grid()
plt.legend()
plt.title("平均损失")
plt.xlabel("训练轮次/epochs")
plt.ylabel("平均损失")
plt.show()
