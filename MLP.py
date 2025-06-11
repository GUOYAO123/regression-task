# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import RobustScaler
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import matplotlib.pyplot as plt
# import os
#
# # 设置字体
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
#
# # 定义数据集类
# class CustomDataset(Dataset):
#     def __init__(self, X, y):
#         self.X = torch.FloatTensor(X)
#         self.y = torch.FloatTensor(y).reshape(-1, 1)
#
#     def __len__(self):
#         return len(self.X)
#
#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]
#
#
# # 定义神经网络模型
# class RegressionNet(nn.Module):
#     def __init__(self, input_dim):
#         super(RegressionNet, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.BatchNorm1d(256),
#             nn.Linear(256, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1)
#         )
#
#     def forward(self, x):
#         return self.net(x)
#
#
# # 加载数据
# data = pd.read_csv(r"F:\English artical\数据\shap数据.csv")
#
# # 选择目标变量和特征
# X = data.drop(columns=['Y'])
# y = data['Y']
#
# # 使用RobustScaler来处理异常值
# scaler = RobustScaler()
# X_scaled = pd.DataFrame(
#     scaler.fit_transform(X),
#     columns=X.columns
# )
#
# # 数据划分：80%训练集，10%验证集，10%测试集
# X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
#
# # 创建数据加载器
# train_dataset = CustomDataset(X_train.values, y_train.values)
# val_dataset = CustomDataset(X_val.values, y_val.values)
# test_dataset = CustomDataset(X_test.values, y_test.values)
#
# train_loader = DataLoader(
#     train_dataset,
#     batch_size=32,
#     shuffle=True,
#     num_workers=0,
#     pin_memory=False
# )
# val_loader = DataLoader(
#     val_dataset,
#     batch_size=32,
#     shuffle=False,
#     num_workers=0,
#     pin_memory=False
# )
# test_loader = DataLoader(
#     test_dataset,
#     batch_size=32,
#     shuffle=False,
#     num_workers=0,
#     pin_memory=False
# )
#
# # 初始化模型
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = RegressionNet(input_dim=len(X.columns)).to(device)
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
# # 训练模型
# num_epochs = 1000
# train_losses = []
# val_losses = []
# best_val_loss = float('inf')
#
# # 创建模型保存路径
# if not os.path.exists('models'):
#     os.makedirs('models')
#
# for epoch in range(num_epochs):
#     model.train()
#     epoch_loss = 0
#     for batch_X, batch_y in train_loader:
#         batch_X, batch_y = batch_X.to(device), batch_y.to(device)
#
#         optimizer.zero_grad()
#         outputs = model(batch_X)
#         loss = criterion(outputs, batch_y)
#         loss.backward()
#         optimizer.step()
#
#         epoch_loss += loss.item()
#
#     train_losses.append(epoch_loss / len(train_loader))
#
#     # 验证
#     model.eval()
#     val_loss = 0
#     with torch.no_grad():
#         for batch_X, batch_y in val_loader:
#             batch_X, batch_y = batch_X.to(device), batch_y.to(device)
#             outputs = model(batch_X)
#             val_loss += criterion(outputs, batch_y).item()
#
#     val_loss = val_loss / len(val_loader)
#     val_losses.append(val_loss)
#
#     # 保存最佳模型
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         torch.save({
#             'model_state_dict': model.state_dict(),
#             'scaler': scaler,
#             'features': X.columns.tolist(),
#         }, 'models/best_model.pth')
#
#     if (epoch + 1) % 10 == 0:
#         print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')
#
# # 保存最终模型
# torch.save({
#     'model_state_dict': model.state_dict(),
#     'scaler': scaler,
#     'features': X.columns.tolist(),
# }, 'models/final_model.pth')
#
# # 加载最佳模型进行测试
# best_model = RegressionNet(input_dim=len(X.columns)).to(device)
# best_model.load_state_dict(torch.load('models/best_model.pth')['model_state_dict'])
# best_model.eval()
#
# # 使用最佳模型进行测试集预测
# y_test_pred = []
# y_test_true = []
#
# with torch.no_grad():
#     for batch_X, batch_y in test_loader:
#         batch_X = batch_X.to(device)
#         outputs = best_model(batch_X)
#         y_test_pred.extend(outputs.cpu().numpy().flatten())
#         y_test_true.extend(batch_y.numpy().flatten())
#
# y_test_true = np.array(y_test_true)
# y_test_pred = np.array(y_test_pred)
#
# # 计算测试集的评估指标
# test_rmse = np.sqrt(((y_test_true - y_test_pred) ** 2).mean())
# test_r2 = 1 - ((y_test_true - y_test_pred) ** 2).sum() / ((y_test_true - y_test_true.mean()) ** 2).sum()
#
# print(f"\n最佳模型在测试集上的表现:")
# print(f"RMSE: {test_rmse:.2f}, R²: {test_r2:.2f}")
#
# # ------------------ 可视化结果 -------------------
#
# # 绘制损失曲线
# plt.figure(figsize=(15, 5))
# plt.subplot(131)
#
# # 过滤异常大的loss值
# max_loss_threshold = np.percentile(train_losses, 95)  # 使用95百分位数作为阈值
# filtered_train_losses = [loss if loss < max_loss_threshold else None for loss in train_losses]
# filtered_val_losses = [loss if loss < max_loss_threshold else None for loss in val_losses]
#
# plt.plot(filtered_train_losses, label='Train Loss')
# plt.plot(filtered_val_losses, label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training and Validation Loss')
# plt.legend()
#
# # 预测值对比图
# plt.subplot(132)
# plt.plot(np.arange(len(y_test_true)), y_test_true, label='True Values', color='blue', linestyle='-', marker='o')
# plt.plot(np.arange(len(y_test_pred)), y_test_pred, label='Predicted Values', color='orange', linestyle='--', marker='x')
# plt.xlabel('Samples')
# plt.ylabel('Values')
# plt.title('Test Set: True vs Predicted Values')
# plt.legend()
#
# # 散点图
# plt.subplot(133)
# plt.scatter(y_test_true, y_test_pred, alpha=0.5)
# plt.plot([y_test_true.min(), y_test_true.max()], [y_test_true.min(), y_test_true.max()], 'r--', lw=2)
# plt.xlabel('True Values')
# plt.ylabel('Predicted Values')
# plt.title('Scatter Plot: True vs Predicted')
#
# plt.tight_layout()
# plt.show()
#
# # 保存预测结果
# results_df = pd.DataFrame({
#     'True_Values': y_test_true,
#     'Predicted_Values': y_test_pred,
#     'Error': y_test_true - y_test_pred
# })
# results_df.to_csv('prediction_results.csv', index=False)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义数据集类
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 定义神经网络模型
class RegressionNet(nn.Module):
    def __init__(self, input_dim):
        super(RegressionNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# 加载数据
# data = pd.read_csv(r"F:\English artical\数据\shap数据.csv")
data = pd.read_csv(r"F:\总样\abs\extracted_columns.csv")
# 获取特征和目标变量
X = data.drop(columns=['Y'])
y = data['Y']

# 使用RobustScaler来处理异常值
scaler = RobustScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 数据划分，首先划分90%训练集和10%测试集
X_train_val, X_test, y_train_val, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

# 在训练集和验证集划分
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.11, random_state=42)


# 创建数据加载器
train_dataset = CustomDataset(X_train.values, y_train.values)
val_dataset = CustomDataset(X_val.values, y_val.values)
test_dataset = CustomDataset(X_test.values, y_test.values)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=False)

# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RegressionNet(input_dim=X_train.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 计算R²和RMSE
def calculate_metrics(true, pred):
    rmse = np.sqrt(((true - pred) ** 2).mean())
    r2 = 1 - ((true - pred) ** 2).sum() / ((true - true.mean()) ** 2).sum()
    return rmse, r2

# 训练模型
num_epochs = 1000
train_losses = []
val_losses = []
test_losses = []
best_val_loss = float('inf')

# 创建模型保存路径
if not os.path.exists('models'):
    os.makedirs('models')

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    train_losses.append(epoch_loss / len(train_loader))

    # 验证
    model.eval()
    val_loss = 0
    val_preds = []
    val_true = []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            val_loss += criterion(outputs, batch_y).item()

            val_preds.extend(outputs.cpu().numpy().flatten())
            val_true.extend(batch_y.cpu().numpy().flatten())

    val_loss = val_loss / len(val_loader)
    val_losses.append(val_loss)

    # 输出验证集的 RMSE 和 R²
    val_rmse, val_r2 = calculate_metrics(np.array(val_true), np.array(val_preds))
    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation RMSE: {val_rmse:.4f}, Validation R²: {val_r2:.4f}')

    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler': scaler,
            'common_features': X.columns.tolist(),
        }, 'models/best_model.pth')

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')

# 保存最终模型
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler': scaler,
    'common_features': X.columns.tolist(),
}, 'models/final_model.pth')

# 加载最佳模型进行预测
best_model = RegressionNet(input_dim=X_train.shape[1]).to(device)
best_model.load_state_dict(torch.load('models/best_model.pth')['model_state_dict'])
best_model.eval()

# 计算测试集的评估指标
y_test_pred = []
y_test_true = []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        outputs = best_model(batch_X)
        y_test_pred.extend(outputs.cpu().numpy().flatten())
        y_test_true.extend(batch_y.numpy().flatten())

y_test_true = np.array(y_test_true)
y_test_pred = np.array(y_test_pred)

# 计算测试集的RMSE和R²
test_rmse, test_r2 = calculate_metrics(y_test_true, y_test_pred)
print(f"\nTest Set RMSE: {test_rmse:.4f}, Test Set R²: {test_r2:.4f}")

# ------------------ 可视化结果 -------------------
import seaborn as sns  # 导入Seaborn库用于数据可视化
colors = sns.color_palette("husl", 3)  # 使用 Seaborn 的调色盘
sns.set_palette("husl", 3)  # 设置颜色盘
plt.figure(figsize=(20, 12), dpi=300)  # 创建画布
plt.tight_layout(pad=2.0, h_pad=3.0, w_pad=2.0)  # 增加调整参数，pad 控制整体边距
# ------------------ 可视化结果 -------------------

# 1. 绘制损失曲线
plt.figure(figsize=(15, 5))
# 过滤异常大的loss值
max_loss_threshold = np.percentile(train_losses, 95)  # 使用95百分位数作为阈值
filtered_train_losses = [loss if loss < max_loss_threshold else None for loss in train_losses]
filtered_val_losses = [loss if loss < max_loss_threshold else None for loss in val_losses]

plt.plot(filtered_train_losses, label='Train Loss',color=colors[0])
plt.plot(filtered_val_losses, label='Validation Loss',color=colors[1])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.tight_layout()
plt.show()

# -------------------------------
# 2. 绘制训练集、验证集和测试集的拟合图
plt.figure(figsize=(15, 5))

# 计算训练集和验证集的预测值
train_preds = []
train_true = []
val_preds = []
val_true = []

# 计算训练集的预测值
model.eval()
with torch.no_grad():
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        outputs = model(batch_X)
        train_preds.extend(outputs.cpu().numpy().flatten())
        train_true.extend(batch_y.numpy().flatten())

# 计算验证集的预测值
with torch.no_grad():
    for batch_X, batch_y in val_loader:
        batch_X = batch_X.to(device)
        outputs = model(batch_X)
        val_preds.extend(outputs.cpu().numpy().flatten())
        val_true.extend(batch_y.numpy().flatten())

# 绘制训练集和验证集拟合图
plt.plot(np.arange(len(train_true)), train_true, label='Train True Values', color='blue', linestyle='-', marker='o')
plt.plot(np.arange(len(train_preds)), train_preds, label='Train Predicted Values', color='cyan', linestyle='--', marker='x')

plt.plot(np.arange(len(val_true)), val_true, label='Validation True Values', color='green', linestyle='-', marker='o')
plt.plot(np.arange(len(val_preds)), val_preds, label='Validation Predicted Values', color='lime', linestyle='--', marker='x')

plt.xlabel('Samples')
plt.ylabel('Values')
plt.title('True vs Predicted Values for Train and Validation Sets')
plt.legend()

plt.tight_layout()
plt.show()

# -------------------------------
# 3. 绘制测试集拟合图
plt.figure(figsize=(15, 5))

# 绘制测试集的拟合图
plt.plot(np.arange(len(y_test_true)), y_test_true, label='Test True Values', color='blue', linestyle='-', marker='o')
plt.plot(np.arange(len(y_test_pred)), y_test_pred, label='Test Predicted Values', color='orange', linestyle='--', marker='x')

plt.xlabel('Samples')
plt.ylabel('Values')
plt.title('True vs Predicted Values for Test Set')
plt.legend()

plt.tight_layout()
plt.show()

# -------------------------------
# 4. 绘制散点图（训练集、验证集、测试集）

import matplotlib.pyplot as plt
import seaborn as sns

# 设置绘图参数
colors = sns.color_palette("husl", 3)  # 使用 Seaborn 的调色盘
sns.set_palette("husl", 3)  # 设置颜色盘
plt.figure(figsize=(20, 12), dpi=300)  # 创建画布
plt.tight_layout(pad=2.0, h_pad=3.0, w_pad=2.0)  # 增加调整参数，pad 控制整体边距

# 绘制训练集散点图
plt.subplot(3, 1, 1)
plt.scatter(train_true, train_preds, alpha=0.3, label='Train Set', color=colors[0])
# plt.plot([train_true.min(), train_true.max()], [train_true.min(), train_true.max()], 'r--', lw=2, label='Ideal Line')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Train Set: True vs Predicted')
plt.legend()

# 绘制验证集散点图
plt.subplot(3, 1, 2)
plt.scatter(val_true, val_preds, alpha=0.3, label='Validation Set', color=colors[1])
# plt.plot([val_true.min(), val_true.max()], [val_true.min(), val_true.max()], 'r--', lw=2, label='Ideal Line')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Validation Set: True vs Predicted')
plt.legend()

# 绘制测试集散点图
plt.subplot(3, 1, 3)
plt.scatter(y_test_true, y_test_pred, alpha=0.3, label='Test Set', color=colors[2])
# plt.plot([y_test_true.min(), y_test_true.max()], [y_test_true.min(), y_test_true.max()], 'r--', lw=2, label='Ideal Line')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Test Set: True vs Predicted')
plt.legend()

# 显示图形
plt.tight_layout()
plt.show()


