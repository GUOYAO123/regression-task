# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签SimHei
# plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
# # plt.rc('font',family='Times New Roman')
#
# # 文件夹路径
# folder_path = r"E:\ZK51\数据一文件\平均化处理文件"
#
# # 获取文件夹中所有CSV文件的路径
# file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]
#
# # 读取所有文件的Smoothed_Reflectance列并存储
# spectra = []
# for file_path in file_paths:
#     df = pd.read_csv(file_path)
#     spectra.append(df['Smoothed_Reflectance'].values)
#
# # 计算均值光谱
# mean_spectrum = np.mean(spectra, axis=0)
#
# # 创建DataFrame保存均值光谱数据
# mean_df = pd.DataFrame({'Wavelength': df['Wavelength'].values, 'Mean Reflectance': mean_spectrum})
#
# # 指定保存路径和文件名7
# output_file_path = r"E:\ZK51\数据一文件\平均化处理文件\h63.csv"
#
# # 保存DataFrame到CSV文件
# mean_df.to_csv(output_file_path, index=False)
#
# # 绘制光谱曲线
# plt.figure(figsize=(10, 6))
# for i, spectrum in enumerate(spectra):
#     plt.plot(df['Wavelength'], spectrum, label=f'Spectrum {i+1}', linestyle='--')
#
# # 绘制均值光谱曲线
# # 设置图的边框宽度
# ax = plt.gca()  # 获取当前的轴
# ax.spines['top'].set_linewidth(2)    # 设置上边框宽度
# ax.spines['right'].set_linewidth(2)  # 设置右边框宽度
# ax.spines['bottom'].set_linewidth(2) # 设置下边框宽度
# ax.spines['left'].set_linewidth(2)   # 设置左边框宽度
# plt.plot(df['Wavelength'], mean_spectrum, label='Mean Spectrum', color='black', linewidth=2)
# plt.xticks(fontsize=24)
# plt.yticks(fontsize=24)
# plt.xlabel('波长（nm）',fontsize=28)
# plt.ylabel('反射率',fontsize=28)
# # plt.title('Average Reflectance Spectrum',fontsize=14)
# plt.legend(fontsize=15)
# # plt.grid(True)
# plt.show()

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 创建字体对象，用于中文字符
chinese_font = FontProperties(fname=r'C:\Windows\Fonts\SimHei.ttf')  # 设置中文字体路径

# 设置字体为Times New Roman
plt.rc('font', family='Times New Roman')

# 文件夹路径
folder_path = r"E:\ZK51\数据一文件\平均化处理文件"

# 获取文件夹中所有CSV文件的路径
file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]

# 读取所有文件的Smoothed_Reflectance列并存储
spectra = []
for file_path in file_paths:
    df = pd.read_csv(file_path)
    spectra.append(df['Smoothed_Reflectance'].values)

# 计算均值光谱
mean_spectrum = np.mean(spectra, axis=0)

# 创建DataFrame保存均值光谱数据
mean_df = pd.DataFrame({'Wavelength': df['Wavelength'].values, 'Mean Reflectance': mean_spectrum})

# 指定保存路径和文件名
output_file_path = r"E:\ZK51\数据一文件\平均化处理文件\h63.csv"

# 保存DataFrame到CSV文件
mean_df.to_csv(output_file_path, index=False)

# 绘制光谱曲线
plt.figure(figsize=(12,6), dpi=300, facecolor='white')  # 设置更大的画布和高DPI，白色背景
colors = plt.cm.tab20.colors  # 使用鲜艳的配色方案（颜色映射）

# 绘制所有光谱
for i, spectrum in enumerate(spectra):
    plt.plot(df['Wavelength'], spectrum, label=f'Spectrum {i+1}', linestyle='-', color=colors[i % len(colors)], alpha=0.7)

# 绘制均值光谱曲线
ax = plt.gca()  # 获取当前的轴
ax.spines['top'].set_linewidth(2)    # 设置上边框宽度
ax.spines['right'].set_linewidth(2)  # 设置右边框宽度
ax.spines['bottom'].set_linewidth(2) # 设置下边框宽度
ax.spines['left'].set_linewidth(2)   # 设置左边框宽度

# 使用加粗的黑色线条表示均值光谱
plt.plot(df['Wavelength'], mean_spectrum, label='Mean Spectrum', color='black', linewidth=3)

# 设置坐标轴字体大小和标签
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('波长（nm）', fontsize=24, fontproperties=chinese_font)
plt.ylabel('反射率', fontsize=24, fontproperties=chinese_font)

# 设置更密集的网格
plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.7, alpha=0.5)  # 主网格
plt.minorticks_on()  # 启用次刻度
plt.grid(True, which='minor', axis='both', linestyle=':', linewidth=0.5, alpha=0.5)  # 次网格，使用细线和点线

# 设置图例
# plt.legend(fontsize=16, loc='upper right', frameon=True, framealpha=0.9, edgecolor='black', facecolor='white')

# 显示图形
plt.tight_layout()  # 自动调整布局
plt.show()


