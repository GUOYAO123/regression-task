# # 三种曲线原始 基于原始的包络线去除  基于基线去除的包络线
from scipy import interpolate
from scipy.spatial import ConvexHull
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
plt.rc('font', family='Arial')  # 使用简洁且常见的字体 Arial

# 读取数据
d = pd.read_csv(r"F:\测试光谱\基线校正后结果.csv")
d1 = pd.read_csv(r"F:\测试光谱\原始csv\30.csv")
x = d.iloc[:, 0].values
y = d.iloc[:, 1].values
x1 = d1.iloc[:, 0].values
y1 = d1.iloc[:, 1].values

# 定义去除包络线的函数
def get_continuum_removal(x, y):
    points = np.array([x, y]).T
    points = points[x.argsort()]
    points = np.vstack([np.array([[points[0, 0], -1]]), points, np.array([[points[-1, 0], -1]])])
    hull = ConvexHull(points)
    temp = hull.vertices
    temp.sort()
    hull = points[temp[1:-1]]
    tck = interpolate.splrep(hull[:, 0], hull[:, 1], k=1)
    iy_hull = interpolate.splev(x, tck, der=0)
    norm = y / iy_hull
    return norm

# 应用包络线去除
norm1 = get_continuum_removal(x, y)
data = np.vstack([x, norm1]).T

norm2 = get_continuum_removal(x1, y1)
data1 = np.vstack([x1, norm2]).T

# 定义误差带的缓冲区
buffer = 0.018

# 绘制图形
plt.figure(figsize=(12, 5))  # 设置图形尺寸

# 绘制去除基线后的光谱，使用鲜艳的橙色
plt.plot(data1[:, 0], data1[:, 1], color='#FF5733', lw=1.5, label='Continuum-Removed Spectrum')
plt.fill_between(data1[:, 0], data1[:, 1] * (1 - buffer), data1[:, 1] * (1 + buffer), color='#FF5733', alpha=0.3)

# 绘制BE去除基线后的光谱，使用紫色
plt.plot(data[:, 0], data[:, 1], color='#8E44AD', lw=1.5, label='BE-Continuum-Removed Spectrum')
plt.fill_between(data[:, 0], data[:, 1] * (1 - buffer), data[:, 1] * (1 + buffer), color='#8E44AD', alpha=0.3)

# 绘制原始光谱，使用深蓝色
plt.plot(x1, y1, color='#1f77b4', lw=1.5, label='Original Spectrum')
plt.fill_between(x1, y1 * (1 - buffer), y1 * (1 + buffer), color='#1f77b4', alpha=0.3)

# 配置图例
plt.legend(frameon=False, loc='upper right', fontsize=12)

# 设置坐标轴标签和标题
plt.xlabel('Wavelength (nm)', fontsize=14)
plt.ylabel('Intensity', fontsize=14)

# 设置坐标轴刻度
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# 显示图形
plt.minorticks_on()  # 启用次刻度
plt.gca().xaxis.set_minor_locator(AutoMinorLocator(5))  # 每5个主刻度之间显示5个次刻度
plt.gca().yaxis.set_minor_locator(AutoMinorLocator(5))  # 每5个主刻度之间显示5个次刻度

# 显示格网（背景格网线，次刻度格网线密集）
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)  # 次刻度和主刻度都显示格网

# 显示图形
plt.tight_layout()  # 使布局更紧凑
plt.show()





# import os
# from scipy import interpolate
# from scipy.spatial import ConvexHull
# import pandas as pd
# import numpy as np
# from scipy.signal import find_peaks
# import matplotlib.pyplot as plt
#
# # Function to perform continuum removal
# def get_continuum_removal(x, y):
#     points = np.array([x, y]).T
#     points = points[x.argsort()]
#     points = np.vstack([np.array([[points[0, 0], -1]]), points, np.array([[points[-1, 0], -1]])])
#     hull = ConvexHull(points)
#     temp = hull.vertices
#     temp.sort()
#     hull = points[temp[1:-1]]
#     tck = interpolate.splrep(hull[:, 0], hull[:, 1], k=1)
#     iy_hull = interpolate.splev(x, tck, der=0)
#     norm = y / iy_hull
#     return norm
#
# # Function to process a single file
# def process_file(file_path, output_folder):
#     # Read the CSV file
#     d = pd.read_csv(file_path)
#     x = d.iloc[:, 0].values
#     y = d.iloc[:, 1].values
#
#     # Apply continuum removal
#     norm = get_continuum_removal(x, y)
#     data = np.vstack([x, norm]).T
#
#     # Get the original file name without extension
#     file_name = os.path.splitext(os.path.basename(file_path))[0]
#
#     # Create a subfolder in the output folder corresponding to the original file's folder
#     file_folder = os.path.basename(os.path.dirname(file_path))
#     subfolder_path = os.path.join(output_folder, file_folder)
#     os.makedirs(subfolder_path, exist_ok=True)
#
#     # Save the continuum-removed spectrum to a new CSV file in the subfolder
#     output_path = os.path.join(subfolder_path, f"{file_name}_continuum_removed.csv")
#     pd.DataFrame(data, columns=['Wavelength', 'Continuum_Removed']).to_csv(output_path, index=False)
#
# # Directory containing CSV files
# # 1input_folder = r"E:\ZK51\柱状图中为未在稀土矿的光谱文件"
# input_folder = r"F:\ZK64-1D&P\数据删除行后数据"
# # Create a new folder to store the processed files
# #1 output_folder = r"E:\ZK51\柱状图中为未在稀土矿的光谱文件baoluoxianquchu"
# output_folder = r"F:\ZK64-1D&P\baoluoxian"
# os.makedirs(output_folder, exist_ok=True)
#
# # Loop through each CSV file in the input folder
# for root, dirs, files in os.walk(input_folder):
#     for file_name in files:
#         if file_name.endswith(".csv"):
#             file_path = os.path.join(root, file_name)
#             process_file(file_path, output_folder)
#
# print("Batch processing completed.")






# 三种曲线原始 基于原始的包络线去除  基于基线去除的包络线


