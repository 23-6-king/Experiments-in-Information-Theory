import numpy as np
import matplotlib.pyplot as plt

def calculate_entropy(p):
    """
    计算给定概率分布p的香农熵。
    参数:
    p (numpy.ndarray): 概率分布向量，其元素之和应为1。
    返回:
    float: 香农熵 (以比特为单位)。
    """
    p = np.array(p)
    # 过滤掉概率为0的项，避免log(0)计算
    p_nonzero = p[p > 0]
    # 根据公式 H(X) = -sum(p * log2(p)) 计算熵
    entropy = -np.sum(p_nonzero * np.log2(p_nonzero))
    return entropy

# --- 主程序 ---
# 任务1: 测试函数
prob_dist_1 = np.array([0.5, 0.5])
entropy_1 = calculate_entropy(prob_dist_1)
print(f"对于概率分布 {prob_dist_1}, 熵为: {entropy_1:.4f} bits")

# 任务2 & 3: 绘制二进制信源熵曲线
p_values = np.linspace(0.01, 0.99, 100)
entropies = []
for p_val in p_values:
    prob_dist = np.array([p_val, 1 - p_val])
    entropies.append(calculate_entropy(prob_dist))

# 任务4: 绘图
plt.figure(figsize=(10, 6))
plt.plot(p_values, entropies, label='H(p)', color='b', linewidth=2)
plt.title('Binary Source Entropy H(p) vs. Probability p')
plt.xlabel('Probability p of symbol "1"')
plt.ylabel('Entropy H(p) (bits/symbol)')
plt.grid(True)
plt.axvline(x=0.5, color='r', linestyle='--', label='Max Entropy at p=0.5')
plt.legend()
plt.show()