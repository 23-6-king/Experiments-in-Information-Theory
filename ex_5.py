import numpy as np

def simulate_bsc(input_seq, p):
    """模拟BSC信道: 以概率p翻转比特"""
    noise = np.random.rand(len(input_seq)) < p
    return np.bitwise_xor(input_seq, noise.astype(int))

# 任务2: 生成随机序列
N = 100000
input_seq = np.random.randint(0, 2, N)

# 任务3: 模拟传输 (p=0.1)
p = 0.1
output_seq = simulate_bsc(input_seq, p)

# 任务4: 计算误码率 (BER)
ber = np.mean(input_seq != output_seq)
print(f"设置错误率 p={p}, 模拟误码率 BER={ber:.5f}")