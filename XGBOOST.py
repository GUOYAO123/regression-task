import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt

# 设置字体为 SimHei（黑体）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据集
# data = pd.read_csv(r"F:\English artical\数据\shap数据.csv")
data = pd.read_csv(r"F:\总样\abs\extracted_columns.csv")
# 提取特征和目标变量
X = data.drop(columns=['Y'])  # 自动提取所有特征列，排除目标变量
y = data['Y']  # 目标变量是 'Y'

# 对目标值进行变换（这里选择对数变换，您可以根据实际情况选择其他变换）
c = 1e-6  # 非常小的正数，以防止log(0)
y_transformed = np.log(y + c)

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 数据分割：首先分割为 80% 训练+验证集 和 20% 测试集
X_train_val, X_test, y_train_val, y_test = train_test_split(X_scaled, y_transformed, test_size=0.2, random_state=42)

# 然后从训练+验证集中分割为 80% 训练集 和 20% 验证集
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# 创建 DMatrix，这是 XGBoost 的专用数据格式
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

# 设置 XGBoost 参数（初始参数，不包含正则化）
params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'min_child_weight': 1,
    'eta': 0.5,
    'subsample': 0.8,
    'colsample_bytree': 1,
    'gamma': 0.25,
    'lambda': 1,
    'alpha': 0,
    'seed': 42,
    'verbosity': 0
}

# 设置网格搜索参数（扩大搜索范围）
param_grid = {
    'max_depth': [3, 5, 7, 10, 15],
    'eta': [0.01, 0.05, 0.1, 0.2, 0.3],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 1.0],
    'lambda': [0, 0.1, 1, 10],
    'alpha': [0, 0.1, 1, 10]
}

# 创建一个包装器以便可以与GridSearchCV一起使用
xgb_model = xgb.XGBRegressor(**params)

# 使用网格搜索来寻找最佳的超参数组合
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)

# 训练模型
grid_search.fit(X_train, y_train)

# 输出最佳参数和得分
print("Best parameters found: ", grid_search.best_params_)
print("Best RMSE score on transformed target: ", np.sqrt(-grid_search.best_score_))

# 使用最佳参数训练最终模型并应用早停法
best_params = grid_search.best_params_
best_params.update(params)

evals_result = {}
evals = [(dtest, 'eval')]
bst = xgb.train(best_params, dtrain, num_boost_round=1000, evals=evals, early_stopping_rounds=100, evals_result=evals_result, verbose_eval=True)

# 预测并逆变换目标变量
def inverse_transform(preds, y_true):
    return np.exp(preds) - c

# 在训练集上预测
y_train_pred_transformed = bst.predict(dtrain)
y_train_pred = inverse_transform(y_train_pred_transformed, y_train)

# 在验证集上预测
y_val_pred_transformed = bst.predict(dval)
y_val_pred = inverse_transform(y_val_pred_transformed, y_val)

# 在测试集上预测
y_test_pred_transformed = bst.predict(dtest)
y_test_pred = inverse_transform(y_test_pred_transformed, y_test)

# 计算训练集、验证集和测试集的 R² 和 RMSE
train_rmse = np.sqrt(mean_squared_error(np.exp(y_train) - c, y_train_pred))
val_rmse = np.sqrt(mean_squared_error(np.exp(y_val) - c, y_val_pred))
test_rmse = np.sqrt(mean_squared_error(np.exp(y_test) - c, y_test_pred))

train_r2 = r2_score(np.exp(y_train) - c, y_train_pred)
val_r2 = r2_score(np.exp(y_val) - c, y_val_pred)
test_r2 = r2_score(np.exp(y_test) - c, y_test_pred)

# 输出每个数据集的 R² 和 RMSE
print(f"训练集 RMSE: {train_rmse:.2f}, R²: {train_r2:.2f}")
print(f"验证集 RMSE: {val_rmse:.2f}, R²: {val_r2:.2f}")
print(f"测试集 RMSE: {test_rmse:.2f}, R²: {test_r2:.2f}")

# 绘制散点图
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
plt.figure(figsize=(18, 12))

# 训练集散点图
plt.subplot(2, 3, 1)
plt.scatter(np.exp(y_train) - c, y_train_pred, color='blue', alpha=0.5)
plt.plot([np.exp(y_train).min() - c, np.exp(y_train).max() - c], [np.exp(y_train).min() - c, np.exp(y_train).max() - c], color='red', linestyle='--', label='Perfect Fit')
plt.title(f"训练集: 真实值 vs 预测值\nRMSE: {train_rmse:.2f}, R²: {train_r2:.2f}")
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.legend()

# 验证集散点图
plt.subplot(2, 3, 2)
plt.scatter(np.exp(y_val) - c, y_val_pred, color='green', alpha=0.5)
plt.plot([np.exp(y_val).min() - c, np.exp(y_val).max() - c], [np.exp(y_val).min() - c, np.exp(y_val).max() - c], color='red', linestyle='--', label='Perfect Fit')
plt.title(f"验证集: 真实值 vs 预测值\nRMSE: {val_rmse:.2f}, R²: {val_r2:.2f}")
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.legend()

# 测试集散点图
plt.subplot(2, 3, 3)
plt.scatter(np.exp(y_test) - c, y_test_pred, color='orange', alpha=0.5)
plt.plot([np.exp(y_test).min() - c, np.exp(y_test).max() - c], [np.exp(y_test).min() - c, np.exp(y_test).max() - c], color='red', linestyle='--', label='Perfect Fit')
plt.title(f"测试集: 真实值 vs 预测值\nRMSE: {test_rmse:.2f}, R²: {test_r2:.2f}")
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.legend()

# 调整布局
plt.tight_layout()

# 显示图表
plt.show()
