import heapq
from collections import Counter
import numpy as np

# 任务1: 定义信源
symbols = ['A', 'B', 'C', 'D', 'E']
probabilities = np.array([0.4, 0.2, 0.2, 0.1, 0.1])

# 任务2: 计算信源熵
def calculate_entropy(p):
    return -np.sum(p * np.log2(p))
print(f"信源熵: {calculate_entropy(probabilities):.4f} bits")

# 任务3: 实现哈夫曼编码
def huffman_coding(symbols, probabilities):
    heap = [[w, [s, ""]] for s, w in zip(symbols, probabilities)]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]: pair[1] = '0' + pair[1]
        for pair in hi[1:]: pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return sorted(heap[0][1:], key=lambda p: (len(p[1]), p[0]))

huff_codes = huffman_coding(symbols, probabilities)
print("哈夫曼码表:", huff_codes)

# 任务4: 计算平均码长
avg_len = sum(probabilities[symbols.index(s)] * len(c) for s, c in huff_codes)
print(f"平均码长: {avg_len:.4f} bits")