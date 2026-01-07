import numpy as np

def calculate_cross_entropy(p_true, p_pred):
    epsilon = 1e-12
    p_pred = np.clip(p_pred, epsilon, 1. - epsilon)
    return -np.sum(p_true * np.log2(p_pred))

# 任务2 & 3: 定义分布
P_true = np.array([0, 1, 0])      # 真实标签(第2类)
Q1 = np.array([0.3, 0.6, 0.1])    # 模型1
Q2 = np.array([0.3, 0.4, 0.3])    # 模型2
Q3 = np.array([0.1, 0.8, 0.1])    # 模型3 (讨论题假设)

# 任务4: 计算
ce1 = calculate_cross_entropy(P_true, Q1)
ce2 = calculate_cross_entropy(P_true, Q2)
ce3 = calculate_cross_entropy(P_true, Q3)

print(f"H(P, Q1) = {ce1:.4f}")
print(f"H(P, Q2) = {ce2:.4f}")
print(f"H(P, Q3) = {ce3:.4f}")