# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# # 精度评估
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#
# plt.rcParams['font.sans-serif'] = 'SimHei'
# plt.rcParams['axes.unicode_minus'] = False
# # df = pd.read_csv("F:\总样\总.csv")
# df = pd.read_csv(r"F:\English artical\数据\shap数据.csv")
# # df = pd.read_csv(r"F:\ZK3-2D&P\融合abs.csv")
# from sklearn.model_selection import train_test_split
#
# X = df.drop(['Y'], axis=1)
# y = df['Y']
#
# X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # 然后将训练集进一步划分为训练集和验证集
# X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
#
# # 数据集标准化
# x_mean = X_train.mean()
# x_std = X_train.std()
# y_mean = y.mean()
# y_std = y.std()
# X_train = (X_train - x_mean) / x_std
# y_train = (y_train - y_mean) / y_std
# X_val = (X_val - x_mean) / x_std
# y_val = (y_val - y_mean) / y_std
# import lightgbm as lgb
#
# # 数据标准化
# X_test = (X_test-x_mean)/x_std
# y_test = (y_test-y_mean) / y_std
#
# # LightGBM模型参数
# params_lgb = {
#     'learning_rate': 0.1,  # 学习率，控制每一步的步长，用于防止过拟合
#     'boosting_type': 'gbdt',  # 提升方法，这里使用梯度提升树（Gradient Boosting Decision Tree）
#     'objective': 'mse',  # 损失函数，均方误差
#     'metric': 'rmse',  # 评估指标，均方根误差
#     'num_leaves': 127,  # 每棵树的叶子节点数量，控制模型复杂度
#     'verbose': -1,  # 控制 LightGBM 输出信息的详细程度，-1 表示不输出
#     'seed': 42,  # 随机种子，用于重现模型的结果
#     'n_jobs': -1,  # 并行运算的线程数量，-1 表示使用所有可用的 CPU
#     'feature_fraction': 0.8,  # 每棵树随机选择的特征比例，用于增加模型的泛化能力
#     'bagging_fraction': 0.9,  # 每次迭代时随机选择的样本比例，用于增加模型的泛化能力
#     'bagging_freq': 4  # 每隔多少次迭代进行一次 bagging 操作，用于减少过拟合
# }
#
# # 创建LightGBM模型
# model_lgb = lgb.LGBMRegressor(**params_lgb)
#
# # 训练模型
# model_lgb.fit(X_train, y_train,
#              eval_set=[(X_val, y_val)],
#              eval_metric='rmse')
# import lightgbm as lgb  # 导入LightGBM库
#
# import seaborn as sns  # 导入Seaborn库用于数据可视化
#
# # 假设你已经训练好了一个LightGBM模型，并准备对训练集、验证集和测试集进行预测
# # 并且你已经计算好了特征的均值和标准差（X_mean, X_std, y_mean, y_std）
#
# # 对训练集、验证集和测试集进行预测
# pred_train = model_lgb.predict(X_train)
# pred_val = model_lgb.predict(X_val)
# pred_test = model_lgb.predict(X_test)
#
# # 将预测结果和真实值还原到原始尺度
# y_train_h = y_train * y_std + y_mean
# pred_train_h = pred_train * y_std + y_mean
# y_val_h = y_val * y_std + y_mean
# pred_val_h = pred_val * y_std + y_mean
# y_test_h = y_test * y_std + y_mean
# pred_test_h = pred_test * y_std + y_mean
#
# # 设置绘图参数
# colors = sns.color_palette("husl", 3)  # 使用 Seaborn 的调色盘
# sns.set_palette("husl", 3)  # 设置颜色盘
# plt.figure(figsize=(20, 12), dpi=300)  # 创建画布
# plt.tight_layout(pad=2.0, h_pad=3.0, w_pad=2.0)  # 增加调整参数，pad 控制整体边距
#
# # 绘制训练集的散点图
# plt.subplot(3, 1, 1)
# plt.scatter(y_train_h, pred_train_h, label='训练集', alpha=0.3, color=colors[0])
# # 计算拟合线
# coeffs = np.polyfit(y_train_h, pred_train_h, 1)  # 一次线性拟合
# fit_line = np.poly1d(coeffs)
# plt.plot(y_train_h, fit_line(y_train_h), color=colors[0], linestyle='-', label='拟合线')
# plt.xlabel('真实值')
# plt.ylabel('预测值')
# # plt.legend()
#
# # 绘制验证集的散点图
# plt.subplot(3, 1, 2)
# plt.scatter(y_val_h, pred_val_h, label='验证集', alpha=0.3, color=colors[1])
# # 计算拟合线
# coeffs = np.polyfit(y_val_h, pred_val_h, 1)
# fit_line = np.poly1d(coeffs)
# plt.plot(y_val_h, fit_line(y_val_h), color=colors[1], linestyle='-', label='拟合线')
# plt.xlabel('真实值')
# plt.ylabel('预测值')
# # plt.legend()
# # ...（如果需要，可以继续绘制测试集的散点图）
# from sklearn.metrics import r2_score
#
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import numpy as np
#
# # 定义一个函数计算多个评价指标
# def calculate_metrics(y_true, y_pred, dataset_name):
#     rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#     mae = mean_absolute_error(y_true, y_pred)
#     r2 = r2_score(y_true, y_pred)
#     print(f"{dataset_name} 评估指标:")
#     print(f"  RMSE: {rmse:.4f}")
#     print(f"  MAE: {mae:.4f}")
#     print(f"  R^2 : {r2:.4f}")
#     print("-" * 30)
#     return rmse, mae, r2
#
# # 还原到原始尺度的训练集、验证集和测试集的指标
# print("训练集指标:")
# calculate_metrics(y_train_h, pred_train_h, "训练集")
#
# print("验证集指标:")
# calculate_metrics(y_val_h, pred_val_h, "验证集")
#
# print("测试集指标:")
# calculate_metrics(y_test_h, pred_test_h, "测试集")
#
# # 训练集拟合线段的R2
# train_r2 = r2_score(y_train_h, fit_line(y_train_h))
# print(f"训练集拟合线的 R^2: {train_r2:.4f}")
#
# # 验证集拟合线段的R2
# val_r2 = r2_score(y_val_h, fit_line(y_val_h))
# print(f"验证集拟合线的 R^2: {val_r2:.4f}")
#
# # 测试集拟合线段的R2
# test_r2 = r2_score(y_test_h, fit_line(y_test_h))
# print(f"测试集拟合线的 R^2: {test_r2:.4f}")
#
# import matplotlib.pyplot as plt  # 导入matplotlib库用于绘图
#
# # 假设你已经准备好了数据：y_test_h（真实值），pred_test_h（预测值），colors（颜色列表）
#
# # 创建一个3行1列的子图，选择第三个子图
# plt.subplot(3, 1, 3)
#
# # 绘制测试集的散点图
# plt.scatter(y_test_h, pred_test_h, label='测试集', alpha=0.3, color=colors[2])
# # 计算拟合线
# coeffs = np.polyfit(y_test_h, pred_test_h, 1)
# fit_line = np.poly1d(coeffs)
# plt.plot(y_test_h, fit_line(y_test_h), color=colors[2], linestyle='-', label='拟合线')
# # 设置坐标轴标签
# plt.xlabel('真实值')
# plt.ylabel('预测值')
#
# # 显示图例
# # plt.legend()
#
#
# plt.tight_layout(pad=2.0, h_pad=3.0, w_pad=2.0)
# plt.show()
# # 显示图形
# plt.show()
#
# import shap
# import matplotlib.pyplot as plt
#
# # 创建SHAP解释器
# explainer = shap.TreeExplainer(model_lgb)
#
# # 计算训练集的SHAP值
# shap_values = explainer.shap_values(X_train)
#
# # 获取特征标签
# labels = X_train.columns
#
# # 设置matplotlib的字体样式
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = 'Times New Roman'
# plt.rcParams['font.size'] = 10  # 调整字体大小
# plt.rcParams['xtick.labelsize'] = 8  # 坐标轴刻度字体
# plt.rcParams['ytick.labelsize'] = 8
#
# # 调整图形大小动态适应特征数量
# plt.figure(figsize=(12, max(6, X_train.shape[1] / 3)))
#
# # 绘制SHAP summary plot，确保展示所有特征
# shap.summary_plot(shap_values, X_train, feature_names=labels, plot_type="dot", max_display=20)
#
#
#
# # 正负shap值
# import shap
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.colors import LinearSegmentedColormap, to_rgba
#
# # 假设 SHAP 计算结果和特征名称已存在
# # 创建SHAP解释器并计算SHAP值
# explainer = shap.TreeExplainer(model_lgb)
# shap_values = explainer.shap_values(X_train)  # SHAP值
# labels = X_train.columns  # 特征标签
#
# # 获取最重要特征对应的SHAP值
# top_shap_values = shap_values  # 如果已经筛选了特征，可以这里筛选
#
# # 计算正贡献和负贡献的平均SHAP值
# mean_positive_shap = np.mean(np.where(top_shap_values > 0, top_shap_values, 0), axis=0)
# mean_negative_shap = np.mean(np.where(top_shap_values < 0, top_shap_values, 0), axis=0)
#
# # 获取正贡献和负贡献前20个特征
# sorted_positive_indices = np.argsort(mean_positive_shap)[::-1][:20]
# sorted_negative_indices = np.argsort(mean_negative_shap)[:20]
#
# positive_features = labels[sorted_positive_indices]
# negative_features = labels[sorted_negative_indices]
#
# # 获取前20个正贡献和负贡献特征对应的SHAP值
# positive_shap_values = mean_positive_shap[sorted_positive_indices]
# negative_shap_values = mean_negative_shap[sorted_negative_indices]
#
# # 定义蓝色和橙色的渐变色
# positive_color = '#007ACC'  # 深蓝色
# negative_color = '#FF6700'  # 橙色
#
# # 创建蓝色渐变色（正贡献）和橙色渐变色（负贡献）
# blue_cmap = LinearSegmentedColormap.from_list("blue_grad", ['#80CFFF', positive_color])  # 蓝色渐变
# orange_cmap = LinearSegmentedColormap.from_list("orange_grad", ['#FFD2A1', negative_color])  # 橙色渐变
#
# # 创建子图
# fig, axes = plt.subplots(2, 1, figsize=(14, 9), dpi=300, gridspec_kw={'height_ratios': [1, 2]})
#
# # **1. 上部分：正贡献前20个特征（使用渐变色）**
# for i, value in enumerate(positive_shap_values):
#     color = blue_cmap(value / max(positive_shap_values))  # 根据SHAP值强度计算渐变色
#     axes[0].barh(i, value, color=color, edgecolor='black')
#
# axes[0].set_yticks(range(len(positive_features)))
# axes[0].set_yticklabels(positive_features, fontsize=10)
# axes[0].set_xlabel("Mean Positive SHAP Value", fontsize=12)
# axes[0].set_title("Top 20 Positive Contribution Features", fontsize=14, fontweight='bold')
# axes[0].grid(axis='x', linestyle='--', alpha=0.5)
#
# # 添加正贡献的颜色条
# norm_pos = plt.Normalize(vmin=min(positive_shap_values), vmax=max(positive_shap_values))
# sm_pos = plt.cm.ScalarMappable(cmap=blue_cmap, norm=norm_pos)
# sm_pos.set_array([])  # 空数组用于ColorBar
# fig.colorbar(sm_pos, ax=axes[0], orientation='vertical', label="SHAP Value")
#
# # **2. 下部分：负贡献前20个特征（使用渐变色）**
# for i, value in enumerate(negative_shap_values):
#     color = orange_cmap(abs(value) / max(abs(negative_shap_values)))  # 根据SHAP值强度计算渐变色
#     axes[1].barh(i, value, color=color, edgecolor='black')
#
# axes[1].set_yticks(range(len(negative_features)))
# axes[1].set_yticklabels(negative_features, fontsize=10)
# axes[1].set_xlabel("Mean Negative SHAP Value", fontsize=12)
# axes[1].set_title("Top 20 Negative Contribution Features", fontsize=14, fontweight='bold')
# axes[1].grid(axis='x', linestyle='--', alpha=0.5)
#
# # 添加负贡献的颜色条
# norm_neg = plt.Normalize(vmin=min(negative_shap_values), vmax=max(negative_shap_values))
# sm_neg = plt.cm.ScalarMappable(cmap=orange_cmap, norm=norm_neg)
# sm_neg.set_array([])  # 空数组用于ColorBar
# fig.colorbar(sm_neg, ax=axes[1], orientation='vertical', label="SHAP Value")
#
# # 调整整体布局，防止标签被遮挡
# plt.tight_layout()
# plt.subplots_adjust(hspace=0.35)  # 增加子图间的间距
# plt.show()
#
#
#
#
# # 训练集可视化
# import matplotlib.pyplot as plt
#
# # 使用您提供代码中已计算的 y_train_h 和 pred_train_h
# x_indices = range(len(y_train_h))  # 使用数据的索引作为横轴
#
# plt.figure(figsize=(10, 6), dpi=300)  # 设置图形大小和分辨率
#
# # 绘制真实值的折线图（Measured）
# plt.plot(x_indices, y_train_h, linestyle='-', color='#9467bd', label='Measured', alpha=0.8)
#
# # 绘制预测值的折线图（Predicted）
# plt.plot(x_indices, pred_train_h, linestyle='-', color='#ffbb78', label='Predicted', alpha=0.8)
#
# # 添加标题和坐标轴标签
# plt.title("Measured vs Predicted (Training Set)", fontsize=14)
# plt.xlabel("Index", fontsize=12)
# plt.ylabel("Value", fontsize=12)
#
# # 显示图例
# plt.legend(fontsize=10)
#
# # 添加网格
# plt.grid(True, linestyle='--', alpha=0.5)
#
# # 显示图形
# plt.tight_layout()
# plt.show()
#
#
# # 测试集可视化
# # 使用您提供代码中已计算的 y_val_h 和 pred_val_h
# y_indices = range(len(y_val_h))  # 使用数据的索引作为横轴
#
# plt.figure(figsize=(10, 6), dpi=300)  # 设置图形大小和分辨率
#
# # 绘制真实值的折线图（Measured）
# plt.plot(y_indices, y_val_h, linestyle='--', color='#9467bd', label='Measured', alpha=0.8)
#
# # 绘制预测值的折线图（Predicted）
# plt.plot(y_indices, pred_val_h, linestyle='-', color='#ffbb78', label='Predicted', alpha=0.8)
#
# # 添加标题和坐标轴标签
# plt.title("Measured vs Predicted (Val Set)", fontsize=14)
# plt.xlabel("Index", fontsize=12)
# plt.ylabel("Value", fontsize=12)
#
# # 显示图例
# plt.legend(fontsize=10)
#
# # 添加网格
# plt.grid(True, linestyle='--', alpha=0.5)
#
# # 显示图形
# plt.tight_layout()
# plt.show()
# #
#
# # 验证集可视化
# # 使用您提供代码中已计算的 y_test_h 和 pred_test_h
# y_indices = range(len(y_test_h))  # 使用数据的索引作为横轴
#
# plt.figure(figsize=(10, 6), dpi=300)  # 设置图形大小和分辨率
#
# # 绘制真实值的折线图（Measured）
# plt.plot(y_indices, y_test_h, linestyle='--', color='#9467bd', label='Measured', alpha=0.8)
#
# # 绘制预测值的折线图（Predicted）
# plt.plot(y_indices, pred_test_h, linestyle='-', color='#ffbb78', label='Predicted', alpha=0.8)
#
# # 添加标题和坐标轴标签
# plt.title("Measured vs Predicted (Test Set)", fontsize=14)
# plt.xlabel("Index", fontsize=12)
# plt.ylabel("Value", fontsize=12)
#
# # 显示图例
# plt.legend(fontsize=10)
#
# # 添加网格
# plt.grid(True, linestyle='--', alpha=0.5)
#
# # 显示图形
# plt.tight_layout()
# plt.show()
#
#
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import lightgbm as lgb
# import seaborn as sns
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.model_selection import train_test_split
# import shap
# from matplotlib.colors import LinearSegmentedColormap  # <-- Add this import
#
# # Set the plot style
# plt.rcParams['font.sans-serif'] = 'SimHei'
# plt.rcParams['axes.unicode_minus'] = False
#
# # Load dataset
# df = pd.read_csv(r"F:\English artical\数据\shap数据.csv")
#
# # Prepare data
# X = df.drop(['Y'], axis=1)
# y = df['Y']
#
# X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
#
# # Data normalization
# x_mean = X_train.mean()
# x_std = X_train.std()
# y_mean = y.mean()
# y_std = y.std()
#
# X_train = (X_train - x_mean) / x_std
# y_train = (y_train - y_mean) / y_std
# X_val = (X_val - x_mean) / x_std
# y_val = (y_val - y_mean) / y_std
# X_test = (X_test - x_mean) / x_std
# y_test = (y_test - y_mean) / y_std
#
# # LightGBM model parameters
# params_lgb = {
#     'learning_rate': 0.1,
#     'boosting_type': 'gbdt',
#     'objective': 'mse',
#     'metric': 'rmse',
#     'num_leaves': 127,
#     'verbose': -1,
#     'seed': 42,
#     'n_jobs': -1,
#     'feature_fraction': 0.8,
#     'bagging_fraction': 0.9,
#     'bagging_freq': 4
# }
#
# # Create LightGBM model
# model_lgb = lgb.LGBMRegressor(**params_lgb)
#
# # Train the model
# model_lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='rmse')
#
# # Predicting
# pred_train = model_lgb.predict(X_train)
# pred_val = model_lgb.predict(X_val)
# pred_test = model_lgb.predict(X_test)
#
# # Reverse scaling
# y_train_h = y_train * y_std + y_mean
# pred_train_h = pred_train * y_std + y_mean
# y_val_h = y_val * y_std + y_mean
# pred_val_h = pred_val * y_std + y_mean
# y_test_h = y_test * y_std + y_mean
# pred_test_h = pred_test * y_std + y_mean
#
# # Set colors for plots
# colors = sns.color_palette("husl", 3)
#
# # Plot layout
# plt.figure(figsize=(18, 15), dpi=300)
# plt.tight_layout(pad=3.0, h_pad=3.0, w_pad=2.0)
#
# # Scatter plot for Training, Validation, and Test sets
# plt.subplot(3, 1, 1)
# plt.scatter(y_train_h, pred_train_h, label='训练集', alpha=0.5, color=colors[0])
# coeffs = np.polyfit(y_train_h, pred_train_h, 1)
# fit_line = np.poly1d(coeffs)
# plt.plot(y_train_h, fit_line(y_train_h), color=colors[0], linestyle='-', label='拟合线')
# plt.xlabel('真实值')
# plt.ylabel('预测值')
#
# plt.subplot(3, 1, 2)
# plt.scatter(y_val_h, pred_val_h, label='验证集', alpha=0.5, color=colors[1])
# coeffs = np.polyfit(y_val_h, pred_val_h, 1)
# fit_line = np.poly1d(coeffs)
# plt.plot(y_val_h, fit_line(y_val_h), color=colors[1], linestyle='-', label='拟合线')
# plt.xlabel('真实值')
# plt.ylabel('预测值')
#
# plt.subplot(3, 1, 3)
# plt.scatter(y_test_h, pred_test_h, label='测试集', alpha=0.5, color=colors[2])
# coeffs = np.polyfit(y_test_h, pred_test_h, 1)
# fit_line = np.poly1d(coeffs)
# plt.plot(y_test_h, fit_line(y_test_h), color=colors[2], linestyle='-', label='拟合线')
# plt.xlabel('真实值')
# plt.ylabel('预测值')
#
# plt.tight_layout(pad=3.0, h_pad=3.0, w_pad=2.0)
# plt.show()
#
# # Evaluate the model
# def calculate_metrics(y_true, y_pred, dataset_name):
#     rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#     mae = mean_absolute_error(y_true, y_pred)
#     r2 = r2_score(y_true, y_pred)
#     print(f"{dataset_name} 评估指标:")
#     print(f"  RMSE: {rmse:.4f}")
#     print(f"  MAE: {mae:.4f}")
#     print(f"  R^2 : {r2:.4f}")
#     print("-" * 30)
#     return rmse, mae, r2
#
# # Print metrics for each dataset
# print("训练集指标:")
# calculate_metrics(y_train_h, pred_train_h, "训练集")
# print("验证集指标:")
# calculate_metrics(y_val_h, pred_val_h, "验证集")
# print("测试集指标:")
# calculate_metrics(y_test_h, pred_test_h, "测试集")
#
# # Create SHAP explainer
# explainer = shap.TreeExplainer(model_lgb)
# shap_values = explainer.shap_values(X_train)
#
# # Plot SHAP summary plot
# plt.figure(figsize=(12, max(6, X_train.shape[1] / 3)))
# shap.summary_plot(shap_values, X_train, feature_names=X_train.columns, plot_type="dot", max_display=20)
#
# # Create positive and negative SHAP value plots
# top_shap_values = shap_values
# mean_positive_shap = np.mean(np.where(top_shap_values > 0, top_shap_values, 0), axis=0)
# mean_negative_shap = np.mean(np.where(top_shap_values < 0, top_shap_values, 0), axis=0)
#
# # sorted_positive_indices = np.argsort(mean_positive_shap)[:20]
# sorted_positive_indices = np.argsort(mean_positive_shap)[::-1][:20]
# sorted_negative_indices = np.argsort(mean_negative_shap)[:20]
#
# positive_features = X_train.columns[sorted_positive_indices]
# negative_features = X_train.columns[sorted_negative_indices]
#
# positive_shap_values = mean_positive_shap[sorted_positive_indices]
# negative_shap_values = mean_negative_shap[sorted_negative_indices]
#
# positive_color = '#007ACC'
# negative_color = '#FF6700'
#
# blue_cmap = LinearSegmentedColormap.from_list("blue_grad", ['#80CFFF', positive_color])
# orange_cmap = LinearSegmentedColormap.from_list("orange_grad", ['#FFD2A1', negative_color])
#
# fig, axes = plt.subplots(2, 1, figsize=(14, 9), dpi=600, gridspec_kw={'height_ratios': [1, 2]})
#
# # Top positive contribution
# for i, value in enumerate(positive_shap_values):
#     color = blue_cmap(value / max(positive_shap_values))
#     axes[0].barh(i, value, color=color, edgecolor='black')
#
# axes[0].set_yticks(range(len(positive_features)))
# axes[0].set_yticklabels(positive_features, fontsize=10)
# axes[0].set_xlabel("Mean Positive SHAP Value", fontsize=12)
# axes[0].set_title("Top 20 Positive Contribution Features", fontsize=14, fontweight='bold')
# axes[0].grid(axis='x', linestyle='--', alpha=0.5)
#
# norm_pos = plt.Normalize(vmin=min(positive_shap_values), vmax=max(positive_shap_values))
# sm_pos = plt.cm.ScalarMappable(cmap=blue_cmap, norm=norm_pos)
# sm_pos.set_array([])
# fig.colorbar(sm_pos, ax=axes[0], orientation='vertical', label="SHAP Value")
#
# # Bottom negative contribution
# for i, value in enumerate(negative_shap_values):
#     color = orange_cmap(abs(value) / max(abs(negative_shap_values)))
#     axes[1].barh(i, value, color=color, edgecolor='black')
#
# axes[1].set_yticks(range(len(negative_features)))
# axes[1].set_yticklabels(negative_features, fontsize=10)
# axes[1].set_xlabel("Mean Negative SHAP Value", fontsize=12)
# axes[1].set_title("Top 20 Negative Contribution Features", fontsize=14, fontweight='bold')
# axes[1].grid(axis='x', linestyle='--', alpha=0.5)
#
# norm_neg = plt.Normalize(vmin=min(negative_shap_values), vmax=max(negative_shap_values))
# sm_neg = plt.cm.ScalarMappable(cmap=orange_cmap, norm=norm_neg)
# sm_neg.set_array([])
# fig.colorbar(sm_neg, ax=axes[1], orientation='vertical', label="SHAP Value")
#
# plt.tight_layout()
# plt.subplots_adjust(hspace=0.35)
# plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import shap
from matplotlib.colors import LinearSegmentedColormap

# Set the plot style
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# Load dataset
df = pd.read_csv(r"F:\English artical\数据\shaply.csv")
# df = pd.read_csv(r"F:\总样\abs\融合abs.csv")
# Prepare data
X = df.drop(['Y'], axis=1)
y = df['Y']

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# Data normalization
x_mean = X_train.mean()
x_std = X_train.std()
y_mean = y.mean()
y_std = y.std()

X_train = (X_train - x_mean) / x_std
y_train = (y_train - y_mean) / y_std
X_val = (X_val - x_mean) / x_std
y_val = (y_val - y_mean) / y_std
X_test = (X_test - x_mean) / x_std
y_test = (y_test - y_mean) / y_std

# LightGBM model parameters
params_lgb = {
    'learning_rate': 0.1,
    'boosting_type': 'gbdt',
    'objective': 'mse',
    'metric': 'rmse',
    'num_leaves': 127,
    'verbose': -1,
    'seed': 42,
    'n_jobs': -1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.9,
    'bagging_freq': 4
}

# Create LightGBM model
model_lgb = lgb.LGBMRegressor(**params_lgb)

# Train the model
model_lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='rmse')

# Predicting
pred_train = model_lgb.predict(X_train)
pred_val = model_lgb.predict(X_val)
pred_test = model_lgb.predict(X_test)

# Reverse scaling
y_train_h = y_train * y_std + y_mean
pred_train_h = pred_train * y_std + y_mean
y_val_h = y_val * y_std + y_mean
pred_val_h = pred_val * y_std + y_mean
y_test_h = y_test * y_std + y_mean
pred_test_h = pred_test * y_std + y_mean

# Set colors for plots
colors = sns.color_palette("husl", 3)

# Plot layout
plt.figure(figsize=(18, 15), dpi=300)
plt.tight_layout(pad=3.0, h_pad=3.0, w_pad=2.0)

# Scatter plot for Training, Validation, and Test sets
plt.subplot(3, 1, 1)
plt.scatter(y_train_h, pred_train_h, label='训练集', alpha=0.5, color=colors[0])
coeffs = np.polyfit(y_train_h, pred_train_h, 1)
fit_line = np.poly1d(coeffs)
plt.plot(y_train_h, fit_line(y_train_h), color=colors[0], linestyle='-', label='拟合线')
plt.xlabel('真实值')
plt.ylabel('预测值')

plt.subplot(3, 1, 2)
plt.scatter(y_val_h, pred_val_h, label='验证集', alpha=0.5, color=colors[1])
coeffs = np.polyfit(y_val_h, pred_val_h, 1)
fit_line = np.poly1d(coeffs)
plt.plot(y_val_h, fit_line(y_val_h), color=colors[1], linestyle='-', label='拟合线')
plt.xlabel('真实值')
plt.ylabel('预测值')

plt.subplot(3, 1, 3)
plt.scatter(y_test_h, pred_test_h, label='测试集', alpha=0.5, color=colors[2])
coeffs = np.polyfit(y_test_h, pred_test_h, 1)
fit_line = np.poly1d(coeffs)
plt.plot(y_test_h, fit_line(y_test_h), color=colors[2], linestyle='-', label='拟合线')
plt.xlabel('真实值')
plt.ylabel('预测值')

plt.tight_layout(pad=3.0, h_pad=3.0, w_pad=2.0)
plt.show()

# Evaluate the model
def calculate_metrics(y_true, y_pred, dataset_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{dataset_name} 评估指标:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R^2 : {r2:.4f}")
    print("-" * 30)
    return rmse, mae, r2

# Print metrics for each dataset
print("训练集指标:")
calculate_metrics(y_train_h, pred_train_h, "训练集")
print("验证集指标:")
calculate_metrics(y_val_h, pred_val_h, "验证集")
print("测试集指标:")
calculate_metrics(y_test_h, pred_test_h, "测试集")

# Create SHAP explainer
explainer = shap.TreeExplainer(model_lgb)
shap_values = explainer.shap_values(X_train)

# Plot SHAP summary plot for top 20 features
plt.figure(figsize=(12, max(6, X_train.shape[1] / 3)))
shap.summary_plot(shap_values, X_train, feature_names=X_train.columns, plot_type="dot", max_display=20)
plt.title("Top 20 Feature Contributions (SHAP)", fontsize=14, fontweight='bold')
plt.show()

# Plot SHAP summary plot for bottom 20 features
# Sort features by mean absolute SHAP values in ascending order
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
bottom_indices = np.argsort(mean_abs_shap)[:20]  # Bottom 20 features
bottom_features = X_train.columns[bottom_indices]
bottom_shap_values = shap_values[:, bottom_indices]

plt.figure(figsize=(12, max(6, len(bottom_features) / 3)))
shap.summary_plot(bottom_shap_values, X_train[bottom_features], feature_names=bottom_features, plot_type="dot", max_display=20)
plt.title("Bottom 20 Feature Contributions (SHAP)", fontsize=14, fontweight='bold')
plt.show()

# Create positive and negative SHAP value plots
top_shap_values = shap_values
mean_positive_shap = np.mean(np.where(top_shap_values > 0, top_shap_values, 0), axis=0)
mean_negative_shap = np.mean(np.where(top_shap_values < 0, top_shap_values, 0), axis=0)

sorted_positive_indices = np.argsort(mean_positive_shap)[::-1][:10]
sorted_negative_indices = np.argsort(mean_negative_shap)[:10]

positive_features = X_train.columns[sorted_positive_indices]
negative_features = X_train.columns[sorted_negative_indices]

positive_shap_values = mean_positive_shap[sorted_positive_indices]
negative_shap_values = mean_negative_shap[sorted_negative_indices]

positive_color = '#007ACC'
negative_color = '#FF6700'

blue_cmap = LinearSegmentedColormap.from_list("blue_grad", ['#80CFFF', positive_color])
orange_cmap = LinearSegmentedColormap.from_list("orange_grad", ['#FFD2A1', negative_color])

fig, axes = plt.subplots(2, 1, figsize=(14, 9), dpi=600, gridspec_kw={'height_ratios': [1, 2]})

# Top positive contribution
for i, value in enumerate(positive_shap_values):
    color = blue_cmap(value / max(positive_shap_values))
    axes[0].barh(i, value, color=color, edgecolor='black')

axes[0].set_yticks(range(len(positive_features)))
axes[0].set_yticklabels(positive_features, fontsize=10)
axes[0].set_xlabel("Mean Positive SHAP Value", fontsize=12)
axes[0].set_title("Top 20 Positive Contribution Features", fontsize=14, fontweight='bold')
axes[0].grid(axis='x', linestyle='--', alpha=0.5)

norm_pos = plt.Normalize(vmin=min(positive_shap_values), vmax=max(positive_shap_values))
sm_pos = plt.cm.ScalarMappable(cmap=blue_cmap, norm=norm_pos)
sm_pos.set_array([])
fig.colorbar(sm_pos, ax=axes[0], orientation='vertical', label="SHAP Value")

# Bottom negative contribution
for i, value in enumerate(negative_shap_values):
    color = orange_cmap(abs(value) / max(abs(negative_shap_values)))
    axes[1].barh(i, value, color=color, edgecolor='black')

axes[1].set_yticks(range(len(negative_features)))
axes[1].set_yticklabels(negative_features, fontsize=10)
axes[1].set_xlabel("Mean Negative SHAP Value", fontsize=12)
axes[1].set_title("Top 20 Negative Contribution Features", fontsize=14, fontweight='bold')
axes[1].grid(axis='x', linestyle='--', alpha=0.5)

norm_neg = plt.Normalize(vmin=min(negative_shap_values), vmax=max(negative_shap_values))
sm_neg = plt.cm.ScalarMappable(cmap=orange_cmap, norm=norm_neg)
sm_neg.set_array([])
fig.colorbar(sm_neg, ax=axes[1], orientation='vertical', label="SHAP Value")

plt.tight_layout()
plt.subplots_adjust(hspace=0.35)
plt.show()