import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer

# 任务1: 加载数据集
wine = load_wine()
X, y = wine.data, wine.target
feature_names = wine.feature_names

# 任务2: 对连续特征进行分箱处理 (互信息计算需要离散变量)
discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
X_discrete = discretizer.fit_transform(X)

# 任务3: 计算互信息
mi_scores = mutual_info_classif(X_discrete, y, discrete_features=True, random_state=42)

# 任务4: 创建DataFrame并排序
mi_series = pd.Series(mi_scores, index=feature_names)
mi_series = mi_series.sort_values(ascending=False)

print("特征互信息排序：")
print(mi_series)

# 任务5: 可视化
plt.figure(figsize=(10, 6))
mi_series.plot(kind='barh', color='teal')
plt.title('Mutual Information Scores for Wine Features')
plt.xlabel('Mutual Information')
plt.show()