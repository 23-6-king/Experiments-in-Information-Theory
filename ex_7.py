import numpy as np
import matplotlib.pyplot as plt

# 任务1: 定义B和SNR范围
B = np.linspace(1e6, 10e6, 50)  # 1MHz - 10MHz
SNR_dB = np.linspace(0, 30, 50) # 0dB - 30dB
SNR = 10 ** (SNR_dB / 10)       # 转换为线性值

B_grid, SNR_grid = np.meshgrid(B, SNR)

# 任务2: 计算容量 C = B * log2(1 + SNR)
C = B_grid * np.log2(1 + SNR_grid)

# 任务3: 绘制3D图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(B_grid/1e6, 10*np.log10(SNR_grid), C/1e6, cmap='viridis')
ax.set_xlabel('Bandwidth B (MHz)')
ax.set_ylabel('SNR (dB)')
ax.set_zlabel('Capacity C (Mbps)')
plt.show()