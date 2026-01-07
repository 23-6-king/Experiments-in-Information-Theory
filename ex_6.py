import numpy as np
import matplotlib.pyplot as plt

def binary_entropy(p):
    if p == 0 or p == 1: return 0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

# 任务2: 设置p范围
p_values = np.linspace(0.001, 0.999, 200)

# 任务3: 计算信道容量 C = 1 - H(p)
capacities = [1 - binary_entropy(p) for p in p_values]

# 任务4: 绘图
plt.figure(figsize=(10, 6))
plt.plot(p_values, capacities, 'r-', linewidth=2)
plt.title('BSC Channel Capacity vs. Error Probability p')
plt.xlabel('Error Probability p')
plt.ylabel('Capacity C (bits/symbol)')
plt.grid(True)
plt.annotate('C=1 (p=0)', xy=(0, 1), xytext=(0.1, 0.9), arrowprops=dict(facecolor='black'))
plt.annotate('C=0 (p=0.5)', xy=(0.5, 0), xytext=(0.5, 0.2), arrowprops=dict(facecolor='black'))
plt.show()

print(f"p=0.1时的容量: {1 - binary_entropy(0.1):.4f}")