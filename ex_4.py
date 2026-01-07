import numpy as np

# 任务1: 定义转移矩阵
P = np.array([[0.7, 0.2, 0.1],
              [0.3, 0.5, 0.2],
              [0.2, 0.4, 0.4]])

# 任务2: 计算平稳分布
def get_stationary_dist(P):
    n = P.shape[0]
    A = np.vstack(((P.T - np.eye(n))[:-1], np.ones(n)))
    b = np.array([0, 0, 1])
    return np.linalg.solve(A, b)

pi = get_stationary_dist(P)
print(f"平稳分布: {pi}")

# 任务3: 计算马尔可夫信源熵
H_markov = -np.sum(pi * np.sum(P * np.log2(P + 1e-12), axis=1))
print(f"马尔可夫信源熵: {H_markov:.4f} bits")

# 比较: 无记忆信源熵
H_memoryless = -np.sum(pi * np.log2(pi + 1e-12))
print(f"同分布无记忆信源熵: {H_memoryless:.4f} bits")