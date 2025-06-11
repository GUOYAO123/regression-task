import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import smogn  # 引入 smogn

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 从 CSV 文件中读取数据
# 假设特征列位于CSV的前n-1列，标签列位于最后一列
data = pd.read_csv(r"F:\总样\abs\Original data.csv")

# 提取特征 (X) 和标签 (y)
X = data.iloc[:, :-1]  # 取前 n-1 列作为特征
y = data.iloc[:, -1]   # 取最后一列作为标签

# 在应用SMOTE算法前打印数据维度
print("应用算法前数据维度:")
print("样本数量:", X.shape[0])
print("特征数量:", X.shape[1])
print("类别分布:", pd.Series(y).value_counts())

# 将特征和标签合并，以符合 smogn 需求
df = pd.concat([X, y], axis=1)
df.columns = [f'feature_{i}' for i in range(X.shape[1])] + ['Y']


# 普遍使用
# # 应用 smogn 进行过采样
df_resampled = smogn.smoter(
    data=df,
    y='Y',
    k=10,  # 增加k值
    samp_method='balance',  # 使用平衡采样
    rel_thres=0.1,  # 降低极端值判定阈值
    rel_method='auto',  # 使用自动相对密度计算方法
    under_samp=True  # 启用欠采样
)

# # 极端过采样
# df_resampled = smogn.smoter(
#     data=df,
#     y='Y',
#     k=5,
#     samp_method='extreme',  # 使用极端采样
#     rel_thres=0.20,  # 调整极端值判定阈值 (值越小, 样本越稀疏)
#     rel_method='auto'  # 使用自动计算相对密度的方法
# )
# 提取过采样后的特征和标签
X_resampled = df_resampled.iloc[:, :-1].values  # 特征
y_resampled = df_resampled.iloc[:, -1].values   # 标签

# 在应用 smogn 过采样算法后打印数据维度
print("\n应用算法后数据维度:")
print("样本数量:", X_resampled.shape[0])
print("特征数量:", X_resampled.shape[1])
print("类别分布:", pd.Series(y_resampled).value_counts())

# 将 smogn 采样后的结果保存到新的 CSV 文件中
resampled_data = np.hstack((X_resampled, y_resampled.reshape(-1, 1)))  # 合并 X 和 y
resampled_df = pd.DataFrame(resampled_data)  # 转换为 DataFrame
resampled_df.to_csv(r"F:\总样\abs\Oversampled data.csv", index=False, header=False)  # 保存为 CSV 文件

# 可视化原始数据分布
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.title("原始数据分布")
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, marker='o', edgecolors='k')
plt.xlabel("特征 1")
plt.ylabel("特征 2")

# 可视化 smogn 过采样后的数据分布
plt.subplot(1, 2, 2)
plt.title("smogn 过采样后数据分布")
plt.scatter(X_resampled[:, 0], X_resampled[:, 1], c=y_resampled, marker='o', edgecolors='k')
plt.xlabel("特征 1")
plt.ylabel("特征 2")

plt.tight_layout()
plt.show()

