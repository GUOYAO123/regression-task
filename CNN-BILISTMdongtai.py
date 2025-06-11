# 原始代码
# import math
# from keras.layers import Conv1D, MaxPooling1D, Flatten, Bidirectional, LSTM, Dense, Dropout
# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# from keras.models import Sequential
# from keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
# from keras.optimizers import Adam
# from sklearn.preprocessing import StandardScaler
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.stats import norm
# import matplotlib
#
# # 设置 Matplotlib 后端和字体
# matplotlib.use('TkAgg')
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
# matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
#
# # 加载数据
# df = pd.read_csv('datalast.csv')
# data = df.values
#
# # 划分训练和测试集
# train_size = 1190
# train_data = data[:train_size]
# test_data = data[train_size:]
#
# window_size = 7
# X_train, y_train = train_data[:, :window_size], train_data[:, window_size]
# X_test, y_test = test_data[:, :window_size], test_data[:, window_size]
#
# # 将输入数据重塑为CNN层的格式
# X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
# X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
#
# # 归一化处理
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train.reshape(-1, window_size)).reshape(X_train.shape)
# X_test = scaler.transform(X_test.reshape(-1, window_size)).reshape(X_test.shape)
#
# # 构建模型
# model_cnn_bilstm = Sequential()
# model_cnn_bilstm.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(window_size, 1)))
# model_cnn_bilstm.add(MaxPooling1D(pool_size=2))
# model_cnn_bilstm.add(Dropout(0.1))
# model_cnn_bilstm.add(Bidirectional(LSTM(64, activation='relu')))
# model_cnn_bilstm.add(Dense(64, activation='relu'))
# model_cnn_bilstm.add(Dropout(0.1))
# model_cnn_bilstm.add(Dense(32, activation='relu'))
# model_cnn_bilstm.add(Dense(1))
#
# # 自定义 Adam 优化器参数
# custom_adam = Adam(learning_rate=0.008, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
#
# def scheduler(epoch, lr):
#     if epoch == 12:
#         return lr * 0.5
#     return lr
#
# def lr_scheduler(epoch, lr):
#     if epoch > 0 and epoch % 20 == 0:
#         return lr * 0.9
#     return lr
#
# # 设置早停和学习率调度器
# early_stopping = EarlyStopping(monitor='val_loss', patience=10)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=2, min_lr=1e-6)
# # reduce_lr = LearningRateScheduler(lr_scheduler)
#
# # 编译模型
# model_cnn_bilstm.compile(optimizer=custom_adam, loss='mse')
#
# # 训练模型
# history = model_cnn_bilstm.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)
# model_cnn_bilstm.summary()
#
# # 可视化损失函数的训练曲线
# # train_loss = history.history['loss']
# # val_loss = history.history['val_loss']
# # epochs = range(1, len(train_loss) + 1)
# # plt.figure(figsize=(20, 8))
# # plt.plot(epochs, train_loss, '#00CED1', label='Training loss')
# # plt.plot(epochs, val_loss, '#8A2BE2', label='Validation loss')
# # plt.title('Training and Validation Loss')
# # plt.xlabel('Epochs')
# # plt.ylabel('Loss')
# # plt.legend()
# # 可视化损失函数的训练曲线
# train_loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(train_loss) + 1)
#
# plt.figure(figsize=(20, 8))
# plt.plot(epochs, train_loss, '#00CED1', label='Training loss')
# plt.plot(epochs, val_loss, '#8A2BE2', label='Validation loss')
#
# # 添加缓冲区：计算损失的最大值和最小值并扩展范围
# min_loss = min(min(train_loss), min(val_loss))
# max_loss = max(max(train_loss), max(val_loss))
# buffer = (max_loss - min_loss) * 0.1  # 添加 10% 的缓冲区
# plt.ylim(min_loss - buffer, max_loss + buffer)  # 设置 y 轴范围
#
# plt.title('Training and Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)  # 可选：添加网格以增强可读性
#
# # 预测
# y_pred = model_cnn_bilstm.predict(X_test)
#
# # 可视化预测结果
# plt.figure(figsize=(20, 8))
# plt.plot(y_test, color='b', label='Actual')
# plt.plot(y_pred, color='r', label='Predicted')
# plt.xlabel('Timestamps')
# plt.ylabel('Values')
# plt.legend()
#
# # 计算评估指标
# mse = mean_squared_error(y_test, y_pred)
# rmse = math.sqrt(mse)
# mae = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
#
# print('MSE: {:.8f}'.format(mse))
# print('RMSE: {:.4f}'.format(rmse))
# print('MAE: {:.4f}'.format(mae))
# print('R^2: {:.4f}'.format(r2))
#
# # 计算残差
# residuals = y_test - y_pred.flatten()
#
# # 绘制残差图
# plt.figure(figsize=(20, 8))
# plt.plot(residuals, 'b-')
# plt.xlabel('检测数据')
# plt.ylabel('残差')
# plt.title('残差图')
#
# # 确定 bin 的数量和范围
# bins = 80
# range_min = np.min(residuals)
# range_max = np.max(residuals)
#
# # 绘制残差直方图和拟合的正态分布曲线
# plt.figure(figsize=(20, 8))
# n, bins, patches = plt.hist(residuals, bins=bins, range=(range_min, range_max), density=True, alpha=0.6, color='b', edgecolor='black')
# mu, std = norm.fit(residuals)  # 拟合正态分布参数
# x = np.linspace(range_min, range_max, 100)
# p = norm.pdf(x, mu, std)
# plt.plot(x, p, 'r-', linewidth=2)
# plt.xlabel('残差')
# plt.ylabel('频数')
# plt.title('残差直方图和正态分布拟合曲线')
#
# # 最后一次性显示所有图像
# plt.show()



#缓冲区


# import math
# from keras.layers import Conv1D, MaxPooling1D, Flatten, Bidirectional, LSTM, Dense, Dropout
# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# from keras.models import Sequential
# from keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
# from keras.optimizers import Adam
# from sklearn.preprocessing import StandardScaler
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.stats import norm
# import matplotlib
#
# # 设置 Matplotlib 后端和字体
# matplotlib.use('TkAgg')
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
# matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
#
# # 加载数据
# df = pd.read_csv('datalast.csv')
# data = df.values
#
# # 划分训练和测试集
# train_size = 1190
# train_data = data[:train_size]
# test_data = data[train_size:]
#
# window_size = 7
# X_train, y_train = train_data[:, :window_size], train_data[:, window_size]
# X_test, y_test = test_data[:, :window_size], test_data[:, window_size]
#
# # 将输入数据重塑为CNN层的格式
# X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
# X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
#
# # 归一化处理
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train.reshape(-1, window_size)).reshape(X_train.shape)
# X_test = scaler.transform(X_test.reshape(-1, window_size)).reshape(X_test.shape)
#
# # 构建模型
# model_cnn_bilstm = Sequential()
# model_cnn_bilstm.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(window_size, 1)))
# model_cnn_bilstm.add(MaxPooling1D(pool_size=2))
# model_cnn_bilstm.add(Dropout(0.1))
# model_cnn_bilstm.add(Bidirectional(LSTM(64, activation='relu')))
# model_cnn_bilstm.add(Dense(64, activation='relu'))
# model_cnn_bilstm.add(Dropout(0.1))
# model_cnn_bilstm.add(Dense(32, activation='relu'))
# model_cnn_bilstm.add(Dense(1))
#
# # 自定义 Adam 优化器参数
# custom_adam = Adam(learning_rate=0.008, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
#
# def scheduler(epoch, lr):
#     if epoch == 12:
#         return lr * 0.5
#     return lr
#
# def lr_scheduler(epoch, lr):
#     if epoch > 0 and epoch % 20 == 0:
#         return lr * 0.9
#     return lr
#
# # 设置早停和学习率调度器
# early_stopping = EarlyStopping(monitor='val_loss', patience=10)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=2, min_lr=1e-6)
#
# # 编译模型
# model_cnn_bilstm.compile(optimizer=custom_adam, loss='mse')
#
# # 训练模型
# history = model_cnn_bilstm.fit(X_train, y_train, epochs=40, batch_size=32, validation_split=0.2)
# model_cnn_bilstm.summary()
#
# # 可视化损失函数的训练曲线（添加缓冲区）
# train_loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(train_loss) + 1)
#
# # 转换为 numpy 数组以便操作
# train_loss = np.array(train_loss)
# val_loss = np.array(val_loss)
#
# plt.figure(figsize=(20, 8))
#
# # 计算缓冲区（这里使用固定比例 10% 的偏移量）
# train_buffer = train_loss * 0.1  # 上下 10% 的范围
# val_buffer = val_loss * 0.1      # 上下 10% 的范围
#
# # 绘制训练损失曲线及其缓冲区
# plt.plot(epochs, train_loss, '#00CED1', label='Training loss')
# plt.fill_between(epochs, train_loss - train_buffer, train_loss + train_buffer, color='#00CED1', alpha=0.4)
#
# # 绘制验证损失曲线及其缓冲区
# plt.plot(epochs, val_loss, '#8A2BE2', label='Validation loss')
# plt.fill_between(epochs, val_loss - val_buffer, val_loss + val_buffer, color='#8A2BE2', alpha=0.4)
#
# plt.title('Training and Validation Loss with Buffer')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)  # 添加网格以增强可读性
#
# # 预测
# y_pred = model_cnn_bilstm.predict(X_test)
#
# # 可视化预测结果
# plt.figure(figsize=(20, 8))
# plt.plot(y_test, color='b', label='Actual')
# plt.plot(y_pred, color='r', label='Predicted')
# plt.xlabel('Timestamps')
# plt.ylabel('Values')
# plt.legend()
#
# # 计算评估指标
# mse = mean_squared_error(y_test, y_pred)
# rmse = math.sqrt(mse)
# mae = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
#
# print('MSE: {:.8f}'.format(mse))
# print('RMSE: {:.4f}'.format(rmse))
# print('MAE: {:.4f}'.format(mae))
# print('R^2: {:.4f}'.format(r2))
#
# # 计算残差
# residuals = y_test - y_pred.flatten()
#
# # 绘制残差图
# plt.figure(figsize=(20, 8))
# plt.plot(residuals, 'b-')
# plt.xlabel('检测数据')
# plt.ylabel('残差')
# plt.title('残差图')
#
# # 确定 bin 的数量和范围
# bins = 80
# range_min = np.min(residuals)
# range_max = np.max(residuals)
#
# # 绘制残差直方图和拟合的正态分布曲线
# plt.figure(figsize=(20, 8))
# n, bins, patches = plt.hist(residuals, bins=bins, range=(range_min, range_max), density=True, alpha=0.6, color='b', edgecolor='black')
# mu, std = norm.fit(residuals)  # 拟合正态分布参数
# x = np.linspace(range_min, range_max, 100)
# p = norm.pdf(x, mu, std)
# plt.plot(x, p, 'r-', linewidth=2)
# plt.xlabel('残差')
# plt.ylabel('频数')
# plt.title('残差直方图和正态分布拟合曲线')
#
# # 最后一次性显示所有图像
# plt.show()

# # 三种数据集的验证
# import math
# from keras.layers import Conv1D, MaxPooling1D, Flatten, Bidirectional, LSTM, Dense, Dropout
# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# from keras.models import Sequential
# from keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from keras.optimizers import Adam
# from sklearn.preprocessing import StandardScaler
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.stats import norm
# import matplotlib
#
# # 设置 Matplotlib 后端和字体
# matplotlib.use('TkAgg')
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# matplotlib.rcParams['axes.unicode_minus'] = False
#
# # 加载数据
# df = pd.read_csv('datalast.csv')
# data = df.values
#
# # 计算数据集大小和划分点 (8:2:2 ratio ≈ 60:20:20)
# total_size = len(data)
# train_size = int(total_size * 0.8)    # 60% for training
# val_size = int(total_size * 0.1)      # 20% for validation
# test_size = total_size - train_size - val_size  # 20% for testing
#
# # 划分训练集、验证集和测试集
# train_data = data[:train_size]
# val_data = data[train_size:train_size + val_size]
# test_data = data[train_size + val_size:]
#
# # 定义窗口大小并准备数据
# window_size = 7
# X_train, y_train = train_data[:, :window_size], train_data[:, window_size]
# X_val, y_val = val_data[:, :window_size], val_data[:, window_size]
# X_test, y_test = test_data[:, :window_size], test_data[:, window_size]
#
# # 重塑输入数据为CNN格式
# X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
# X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
# X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
#
# # 归一化处理
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train.reshape(-1, window_size)).reshape(X_train.shape)
# X_val = scaler.transform(X_val.reshape(-1, window_size)).reshape(X_val.shape)
# X_test = scaler.transform(X_test.reshape(-1, window_size)).reshape(X_test.shape)
#
# # 构建模型
# model_cnn_bilstm = Sequential()
# model_cnn_bilstm.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(window_size, 1)))
# model_cnn_bilstm.add(MaxPooling1D(pool_size=2))
# model_cnn_bilstm.add(Dropout(0.1))
# model_cnn_bilstm.add(Bidirectional(LSTM(64, activation='relu')))
# model_cnn_bilstm.add(Dense(64, activation='relu'))
# model_cnn_bilstm.add(Dropout(0.1))
# model_cnn_bilstm.add(Dense(32, activation='relu'))
# model_cnn_bilstm.add(Dense(1))
#
# # 自定义优化器
# custom_adam = Adam(learning_rate=0.008, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
#
# # 设置回调函数
# early_stopping = EarlyStopping(monitor='val_loss', patience=10)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=2, min_lr=1e-6)
# # 编译模型
# model_cnn_bilstm.compile(optimizer=custom_adam, loss='mse')
#
# # 训练模型
# history = model_cnn_bilstm.fit(X_train, y_train, epochs=500, batch_size=32,
#                              validation_data=(X_val, y_val),
#                              callbacks=[early_stopping, reduce_lr])
# # history = model_cnn_bilstm.fit(X_train, y_train, epochs=40, batch_size=32, validation_split=0.2)
# model_cnn_bilstm.summary()
#
# # 可视化训练过程中的损失
# plt.figure(figsize=(20, 8))
# train_loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(train_loss) + 1)
#
# train_loss = np.array(train_loss)
# val_loss = np.array(val_loss)
#
# train_buffer = train_loss * 0.1
# val_buffer = val_loss * 0.1
#
# plt.plot(epochs, train_loss, '#00CED1', label='Training loss')
# plt.fill_between(epochs, train_loss - train_buffer, train_loss + train_buffer, color='#00CED1', alpha=0.2)
# plt.plot(epochs, val_loss, '#8A2BE2', label='Validation loss')
# plt.fill_between(epochs, val_loss - val_buffer, val_loss + val_buffer, color='#8A2BE2', alpha=0.2)
# plt.title('Training and Validation Loss with Buffer')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)
#
# # 在验证集上进行预测和评估
# y_val_pred = model_cnn_bilstm.predict(X_val)
#
# # 可视化验证集预测结果
# plt.figure(figsize=(20, 8))
# plt.plot(y_val, color='b', label='Actual Validation')
# plt.plot(y_val_pred, color='r', label='Predicted Validation')
# plt.xlabel('Timestamps')
# plt.ylabel('Values')
# plt.title('Validation Set: Actual vs Predicted')
# plt.legend()
#
# # 计算验证集评估指标
# val_mse = mean_squared_error(y_val, y_val_pred)
# val_rmse = math.sqrt(val_mse)
# val_mae = mean_absolute_error(y_val, y_val_pred)
# val_r2 = r2_score(y_val, y_val_pred)
#
# print('\nValidation Set Metrics:')
# print('MSE: {:.8f}'.format(val_mse))
# print('RMSE: {:.4f}'.format(val_rmse))
# print('MAE: {:.4f}'.format(val_mae))
# print('R^2: {:.4f}'.format(val_r2))
#
# # 在测试集上进行预测和评估
# y_test_pred = model_cnn_bilstm.predict(X_test)
#
# # 可视化测试集预测结果
# plt.figure(figsize=(20, 8))
# plt.plot(y_test, color='b', label='Actual Test')
# plt.plot(y_test_pred, color='r', label='Predicted Test')
# plt.xlabel('Timestamps')
# plt.ylabel('Values')
# plt.title('Test Set: Actual vs Predicted')
# plt.legend()
#
# # 计算测试集评估指标
# test_mse = mean_squared_error(y_test, y_test_pred)
# test_rmse = math.sqrt(test_mse)
# test_mae = mean_absolute_error(y_test, y_test_pred)
# test_r2 = r2_score(y_test, y_test_pred)
#
# print('\nTest Set Metrics:')
# print('MSE: {:.8f}'.format(test_mse))
# print('RMSE: {:.4f}'.format(test_rmse))
# print('MAE: {:.4f}'.format(test_mae))
# print('R^2: {:.4f}'.format(test_r2))
#
# # 计算测试集残差
# residuals = y_test - y_test_pred.flatten()
#
# # 绘制测试集残差图
# plt.figure(figsize=(20, 8))
# plt.plot(residuals, 'b-')
# plt.xlabel('检测数据')
# plt.ylabel('残差')
# plt.title('测试集残差图')
#
# # 绘制测试集残差直方图和正态分布曲线
# plt.figure(figsize=(20, 8))
# bins = 80
# range_min = np.min(residuals)
# range_max = np.max(residuals)
# n, bins, patches = plt.hist(residuals, bins=bins, range=(range_min, range_max), density=True, alpha=0.6, color='b', edgecolor='black')
# mu, std = norm.fit(residuals)
# x = np.linspace(range_min, range_max, 100)
# p = norm.pdf(x, mu, std)
# plt.plot(x, p, 'r-', linewidth=2)
# plt.xlabel('残差')
# plt.ylabel('频数')
# plt.title('测试集残差直方图和正态分布拟合曲线')
#
# plt.show()



# # 上面版本与此均可以
# import math
# from keras.layers import Conv1D, MaxPooling1D, Flatten, Bidirectional, LSTM, Dense, Dropout,BatchNormalization
# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# from keras.models import Sequential
# from keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from keras.optimizers import Adam
# from sklearn.preprocessing import StandardScaler,MinMaxScaler
# from keras import regularizers
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.stats import norm
# import matplotlib
#
# # 设置 Matplotlib 后端和字体
# matplotlib.use('TkAgg')
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# matplotlib.rcParams['axes.unicode_minus'] = False
#
# # 加载数据
# df = pd.read_csv("extracted_columns.csv")
# # df = pd.read_csv("datalast.csv")
# data = df.values
#
# print("NaN in raw data:", np.any(np.isnan(data)))
# #
# # 计算数据集大小和划分点 (8:2:2 ratio ≈ 60:20:20)
# total_size = len(data)
# train_size = int(total_size * 0.8)    # 60% for training
# val_size = int(total_size * 0.1)      # 20% for validation
# test_size = total_size - train_size - val_size  # 20% for testing
#
# # 划分训练集、验证集和测试集
# train_data = data[:train_size]
# val_data = data[train_size:train_size + val_size]
# test_data = data[train_size + val_size:]
#
# # 定义窗口大小并准备数据
# window_size = 24
# X_train, y_train = train_data[:, :window_size], train_data[:, window_size]
# X_val, y_val = val_data[:, :window_size], val_data[:, window_size]
# X_test, y_test = test_data[:, :window_size], test_data[:, window_size]
#
# # 打印目标值的统计信息以诊断问题
# print("y_train statistics:")
# print("Mean:", np.mean(y_train))
# print("Std:", np.std(y_train))
# print("Min:", np.min(y_train))
# print("Max:", np.max(y_train))
#
# # 对目标值进行归一化
# scaler_y = StandardScaler()
# y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
# y_val = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
# y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
#
# # # 重塑输入数据为CNN格式
# X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
# X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
# X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
#
# # 归一化输入数据
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train.reshape(-1, window_size)).reshape(X_train.shape)
# X_val = scaler.transform(X_val.reshape(-1, window_size)).reshape(X_val.shape)
# X_test = scaler.transform(X_test.reshape(-1, window_size)).reshape(X_test.shape)
#
# # 构建模型
# # model_cnn_bilstm = Sequential()
# # model_cnn_bilstm.add(Conv1D(filters=128, kernel_size=5, activation='relu', input_shape=(window_size, 1)))
# # model_cnn_bilstm.add(BatchNormalization())
# # model_cnn_bilstm.add(MaxPooling1D(pool_size=2))
# # model_cnn_bilstm.add(Dropout(0.3))
# #
# # model_cnn_bilstm.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
# # model_cnn_bilstm.add(BatchNormalization())
# # model_cnn_bilstm.add(MaxPooling1D(pool_size=2))
# # model_cnn_bilstm.add(Dropout(0.3))
# #
# # model_cnn_bilstm.add(Bidirectional(LSTM(128, activation='relu', return_sequences=True)))
# # model_cnn_bilstm.add(Dropout(0.3))
# # model_cnn_bilstm.add(Bidirectional(LSTM(64, activation='relu')))
# #
# # model_cnn_bilstm.add(Dense(128, activation='relu'))
# # model_cnn_bilstm.add(Dropout(0.3))
# # model_cnn_bilstm.add(Dense(64, activation='relu'))
# # model_cnn_bilstm.add(Dense(1))
#
#
# # 添加L2正则化技术
# model_cnn_bilstm = Sequential()
# model_cnn_bilstm.add(Conv1D(filters=128, kernel_size=5, activation='relu', input_shape=(window_size, 1),
#                              kernel_regularizer=regularizers.l2(0.001)))  # 添加 L2 正则化
# model_cnn_bilstm.add(BatchNormalization())
# model_cnn_bilstm.add(MaxPooling1D(pool_size=2))
# model_cnn_bilstm.add(Dropout(0.3))
#
# model_cnn_bilstm.add(Conv1D(filters=64, kernel_size=3, activation='relu',
#                              kernel_regularizer=regularizers.l2(0.001)))  # 添加 L2 正则化
# model_cnn_bilstm.add(BatchNormalization())
# model_cnn_bilstm.add(MaxPooling1D(pool_size=2))
# model_cnn_bilstm.add(Dropout(0.3))
#
# model_cnn_bilstm.add(Bidirectional(LSTM(128, activation='relu', return_sequences=True,
#                                         kernel_regularizer=regularizers.l2(0.001))))  # 添加 L2 正则化
# model_cnn_bilstm.add(Dropout(0.3))
# model_cnn_bilstm.add(Bidirectional(LSTM(64, activation='relu',
#                                         kernel_regularizer=regularizers.l2(0.001))))  # 添加 L2 正则化
#
# model_cnn_bilstm.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))  # 添加 L2 正则化
# model_cnn_bilstm.add(Dropout(0.3))
# model_cnn_bilstm.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))  # 添加 L2 正则化
# model_cnn_bilstm.add(Dense(1))
#
# custom_adam = Adam(learning_rate=0.0005)
#
# early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-8)
#
# model_cnn_bilstm.compile(optimizer=custom_adam, loss='mse')
#
# history = model_cnn_bilstm.fit(X_train, y_train, epochs=400, batch_size=64,
#                                validation_data=(X_val, y_val),
#                                callbacks=[early_stopping, reduce_lr])
# model_cnn_bilstm.summary()
#
#
# # 可视化训练过程中的损失
# plt.figure(figsize=(20, 8))
# train_loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(train_loss) + 1)
#
# train_loss = np.array(train_loss)
# val_loss = np.array(val_loss)
#
# train_buffer = train_loss * 0.1
# val_buffer = val_loss * 0.1
#
# plt.plot(epochs, train_loss, '#00CED1', label='Training loss')
# plt.fill_between(epochs, train_loss - train_buffer, train_loss + train_buffer, color='#00CED1', alpha=0.2)
# plt.plot(epochs, val_loss, '#8A2BE2', label='Validation loss')
# plt.fill_between(epochs, val_loss - val_buffer, val_loss + val_buffer, color='#8A2BE2', alpha=0.2)
# plt.title('Training and Validation Loss with Buffer')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)
#
# # 在验证集上进行预测和评估
# y_val_pred = model_cnn_bilstm.predict(X_val)
# y_val_pred = scaler_y.inverse_transform(y_val_pred)  # 逆变换到原始尺度
# y_val = scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()
#
# # 可视化验证集预测结果
# plt.figure(figsize=(20, 8))
# plt.plot(y_val, color='b', label='Actual Validation')
# plt.plot(y_val_pred, color='r', label='Predicted Validation')
# plt.xlabel('Timestamps')
# plt.ylabel('Values')
# plt.title('Validation Set: Actual vs Predicted')
# plt.legend()
#
# # 计算验证集评估指标
# val_mse = mean_squared_error(y_val, y_val_pred)
# val_rmse = math.sqrt(val_mse)
# val_mae = mean_absolute_error(y_val, y_val_pred)
# val_r2 = r2_score(y_val, y_val_pred)
#
# print('\nValidation Set Metrics:')
# print('MSE: {:.8f}'.format(val_mse))
# print('RMSE: {:.4f}'.format(val_rmse))
# print('MAE: {:.4f}'.format(val_mae))
# print('R^2: {:.4f}'.format(val_r2))
#
# # 在测试集上进行预测和评估
# y_test_pred = model_cnn_bilstm.predict(X_test)
# y_test_pred = scaler_y.inverse_transform(y_test_pred)  # 逆变换到原始尺度
# y_test = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
#
# # 可视化测试集预测结果
# plt.figure(figsize=(20, 8))
# plt.plot(y_test, color='b', label='Actual Test')
# plt.plot(y_test_pred, color='r', label='Predicted Test')
# plt.xlabel('Timestamps')
# plt.ylabel('Values')
# plt.title('Test Set: Actual vs Predicted')
# plt.legend()
#
# # 计算测试集评估指标
# test_mse = mean_squared_error(y_test, y_test_pred)
# test_rmse = math.sqrt(test_mse)
# test_mae = mean_absolute_error(y_test, y_test_pred)
# test_r2 = r2_score(y_test, y_test_pred)
#
# print('\nTest Set Metrics:')
# print('MSE: {:.8f}'.format(test_mse))
# print('RMSE: {:.4f}'.format(test_rmse))
# print('MAE: {:.4f}'.format(test_mae))
# print('R^2: {:.4f}'.format(test_r2))
#
# # 计算测试集残差
# residuals = y_test - y_test_pred.flatten()
#
# # 绘制测试集残差图
# plt.figure(figsize=(20, 8))
# plt.plot(residuals, 'b-')
# plt.xlabel('检测数据')
# plt.ylabel('残差')
# plt.title('测试集残差图')
#
# # 绘制测试集残差直方图和正态分布曲线
# plt.figure(figsize=(20, 8))
# bins = 80
# range_min = np.min(residuals)
# range_max = np.max(residuals)
# n, bins, patches = plt.hist(residuals, bins=bins, range=(range_min, range_max), density=True, alpha=0.6, color='b', edgecolor='black')
# mu, std = norm.fit(residuals)
# x = np.linspace(range_min, range_max, 100)
# p = norm.pdf(x, mu, std)
# plt.plot(x, p, 'r-', linewidth=2)
# plt.xlabel('残差')
# plt.ylabel('频数')
# plt.title('测试集残差直方图和正态分布拟合曲线')
#
# plt.show()








# select feature
# feature_cols=["X560","X580","X624","X590","X734","X740","X750","X680","X795","X805","X865","X1714","X2207",
#                 "X2342","X524","X512","X1493","X1908","X1464","X1550","X2152","X888","X1363","X1375"]
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, probplot
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
from keras import regularizers
import matplotlib
import warnings
import keras_tuner as kt

# 设置 Matplotlib 字体和后端
try:
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 优先使用 SimHei 字体
except:
    matplotlib.rcParams['font.sans-serif'] = ['Arial']  # 后备字体
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.use('Agg')  # 使用非交互式后端

# 忽略警告
warnings.filterwarnings('ignore')


# 数据预处理函数
def preprocess_data(file_path, window_size=24, train_ratio=0.8, val_ratio=0.1):
    """
    预处理时间序列数据，包括加载、分割、窗口化和归一化。
    返回训练、验证、测试数据和归一化器。
    """
    df = pd.read_csv(file_path)
    if df.isna().sum().sum() > 0:
        raise ValueError("数据中存在缺失值，请检查！")
    if not np.all(df.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
        raise ValueError("数据中存在非数值类型，请检查！")

    data = df.values
    print("数据形状:", data.shape)
    print("目标值统计 - 均值: {:.4f}, 标准差: {:.4f}, 最小值: {:.4f}, 最大值: {:.4f}".format(
        np.mean(data[:, -1]), np.std(data[:, -1]), np.min(data[:, -1]), np.max(data[:, -1])))

    # 数据分割
    total_size = len(data)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    # 窗口化
    X_train, y_train = train_data[:, :window_size], train_data[:, -1]
    X_val, y_val = val_data[:, :window_size], val_data[:, -1]
    X_test, y_test = test_data[:, :window_size], test_data[:, -1]

    # 归一化
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    scaler_x = StandardScaler()
    X_train = scaler_x.fit_transform(X_train).reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = scaler_x.transform(X_val).reshape(X_val.shape[0], X_val.shape[1], 1)
    X_test = scaler_x.transform(X_test).reshape(X_test.shape[0], X_test.shape[1], 1)

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler_x, scaler_y


# 构建模型函数
def build_cnn_bilstm_model(window_size, l2_reg=0.001, dropout_rate=0.2, filters_1=64, filters_2=32, lstm_units_1=64,
                           lstm_units_2=32):
    """
    构建 CNN-BiLSTM 模型，优化激活函数和丢弃率。
    """
    model = Sequential()
    model.add(Conv1D(filters=filters_1, kernel_size=5, activation='relu', input_shape=(window_size, 1),
                     kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(dropout_rate))

    model.add(Conv1D(filters=filters_2, kernel_size=3, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(dropout_rate))

    model.add(Bidirectional(LSTM(lstm_units_1, activation='tanh', return_sequences=True,
                                 kernel_regularizer=regularizers.l2(l2_reg))))
    model.add(Dropout(dropout_rate))
    model.add(Bidirectional(LSTM(lstm_units_2, activation='tanh',
                                 kernel_regularizer=regularizers.l2(l2_reg))))

    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(Dense(1))

    return model


# 超参数调优模型
def build_tuner_model(hp):
    """
    为 keras-tuner 构建模型，搜索超参数。
    """
    model = Sequential()
    model.add(Conv1D(
        filters=hp.Int('filters_1', 32, 128, step=32),
        kernel_size=hp.Choice('kernel_size_1', [3, 5]),
        activation='relu',
        input_shape=(hp.Int('window_size', 12, 48, step=12), 1),
        kernel_regularizer=regularizers.l2(hp.Float('l2', 1e-4, 1e-2, sampling='log'))
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(hp.Float('dropout', 0.1, 0.3, step=0.1)))

    model.add(Conv1D(
        filters=hp.Int('filters_2', 16, 64, step=16),
        kernel_size=hp.Choice('kernel_size_2', [3, 5]),
        activation='relu',
        kernel_regularizer=regularizers.l2(hp.Float('l2', 1e-4, 1e-2, sampling='log'))
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(hp.Float('dropout', 0.1, 0.3, step=0.1)))

    model.add(Bidirectional(LSTM(
        units=hp.Int('lstm_units_1', 32, 128, step=32),
        activation='tanh',
        return_sequences=True,
        kernel_regularizer=regularizers.l2(hp.Float('l2', 1e-4, 1e-2, sampling='log'))
    )))
    model.add(Dropout(hp.Float('dropout', 0.1, 0.3, step=0.1)))
    model.add(Bidirectional(LSTM(
        units=hp.Int('lstm_units_2', 16, 64, step=16),
        activation='tanh',
        kernel_regularizer=regularizers.l2(hp.Float('l2', 1e-4, 1e-2, sampling='log'))
    )))

    model.add(Dense(
        units=hp.Int('dense_units', 32, 128, step=32),
        activation='relu',
        kernel_regularizer=regularizers.l2(hp.Float('l2', 1e-4, 1e-2, sampling='log'))
    ))
    model.add(Dropout(hp.Float('dropout', 0.1, 0.3, step=0.1)))
    model.add(Dense(1))

    model.compile(optimizer=Adam(hp.Float('lr', 1e-4, 1e-2, sampling='log')), loss='huber')
    return model


# 评估模型函数
def evaluate_model(model, X, y, scaler_y, dataset_name="Validation"):
    """
    评估模型性能，计算 MSE、RMSE、MAE、MAPE 和 R²。
    """
    y_pred = model.predict(X, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred).flatten()
    y_true = scaler_y.inverse_transform(y.reshape(-1, 1)).flatten()

    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{dataset_name} 集评估指标:")
    print(f"MSE: {mse:.8f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.4f}")
    print(f"R²: {r2:.4f}")

    return y_true, y_pred, {"mse": mse, "rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


# 可视化函数
def plot_results(history, y_val_true, y_val_pred, y_test_true, y_test_pred, residuals):
    """
    绘制训练损失、预测结果和残差分析。
    """
    # 训练和验证损失
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='训练损失', color='#00CED1')
    plt.plot(history.history['val_loss'], label='验证损失', color='#8A2BE2')
    plt.title('训练和验证损失')
    plt.xlabel('Epochs')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png')
    plt.close()

    # 验证集预测
    plt.figure(figsize=(12, 6))
    plt.plot(y_val_true, label='实际值', color='blue')
    plt.plot(y_val_pred, label='预测值', color='red')
    plt.title('验证集：实际值 vs 预测值')
    plt.xlabel('时间步')
    plt.ylabel('值')
    plt.legend()
    plt.grid(True)
    plt.savefig('val_pred_plot.png')
    plt.close()

    # 测试集预测
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_true, label='实际值', color='blue')
    plt.plot(y_test_pred, label='预测值', color='red')
    plt.title('测试集：实际值 vs 预测值')
    plt.xlabel('时间步')
    plt.ylabel('值')
    plt.legend()
    plt.grid(True)
    plt.savefig('test_pred_plot.png')
    plt.close()

    # 残差图
    plt.figure(figsize=(12, 6))
    plt.plot(residuals, label='残差', color='blue')
    plt.title('测试集残差图')
    plt.xlabel('时间步')
    plt.ylabel('残差')
    plt.legend()
    plt.grid(True)
    plt.savefig('residuals_plot.png')
    plt.close()

    # 残差直方图和正态分布
    plt.figure(figsize=(12, 6))
    n, bins, patches = plt.hist(residuals, bins=50, density=True, alpha=0.6, color='blue', edgecolor='black')
    mu, std = norm.fit(residuals)
    x = np.linspace(min(residuals), max(residuals), 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'r-', linewidth=2, label='正态分布拟合')
    plt.title('测试集残差直方图和正态分布拟合')
    plt.xlabel('残差')
    plt.ylabel('密度')
    plt.legend()
    plt.grid(True)
    plt.savefig('residuals_hist_plot.png')
    plt.close()

    # 残差 Q-Q 图
    plt.figure(figsize=(12, 6))
    probplot(residuals, dist="norm", plot=plt)
    plt.title('测试集残差 Q-Q 图')
    plt.savefig('residuals_qq_plot.png')
    plt.close()


# 时间序列交叉验证
def cross_validate_model(file_path, window_size=24, n_splits=5):
    """
    使用 TimeSeriesSplit 进行交叉验证。
    """
    df = pd.read_csv(file_path)
    data = df.values
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(data), 1):
        print(f"\n交叉验证折 {fold}")
        train_data = data[train_idx]
        val_data = data[val_idx]
        X_train, y_train = train_data[:, :window_size], train_data[:, -1]
        X_val, y_val = val_data[:, :window_size], val_data[:, -1]

        scaler_y = StandardScaler()
        y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val = scaler_y.transform(y_val.reshape(-1, 1)).flatten()

        scaler_x = StandardScaler()
        X_train = scaler_x.fit_transform(X_train).reshape(X_train.shape[0], X_train.shape[1], 1)
        X_val = scaler_x.transform(X_val).reshape(X_val.shape[0], X_val.shape[1], 1)

        model = build_cnn_bilstm_model(window_size)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='huber')

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
            ModelCheckpoint(f'best_model_fold_{fold}.keras', monitor='val_loss', save_best_only=True)
        ]

        model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val),
                  callbacks=callbacks, verbose=1)

        _, _, val_metrics = evaluate_model(model, X_val, y_val, scaler_y, f"交叉验证折 {fold}")
        metrics.append(val_metrics)

    avg_metrics = {key: np.mean([m[key] for m in metrics]) for key in metrics[0]}
    print("\n平均交叉验证指标:", avg_metrics)
    return avg_metrics


# 主函数
def main():
    # 参数配置
    file_path = "extracted_columns.csv"
    # file_path = "融合abs.csv"
    window_size = 24
    train_ratio = 0.8
    val_ratio = 0.1
    epochs = 200
    batch_size = 32
    n_splits = 5
    use_tuner = False  # 是否使用超参数调优

    # 数据预处理
    X_train, y_train, X_val, y_val, X_test, y_test, scaler_x, scaler_y = preprocess_data(
        file_path, window_size, train_ratio, val_ratio)

    if use_tuner:
        # 超参数调优
        tuner = kt.Hyperband(
            build_tuner_model,
            objective='val_loss',
            max_epochs=100,
            factor=3,
            directory='tuner_dir',
            project_name='cnn_bilstm_tuning'
        )
        tuner.search(X_train, y_train, epochs=50, validation_data=(X_val, y_val),
                     callbacks=[EarlyStopping(monitor='val_loss', patience=10)])
        best_model = tuner.get_best_models(num_models=1)[0]
        print("最佳超参数:", tuner.get_best_hyperparameters()[0].values)
    else:
        # 使用默认模型
        best_model = build_cnn_bilstm_model(window_size)
        best_model.compile(optimizer=Adam(learning_rate=0.001), loss='huber')

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
            ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
        ]

        history = best_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                                 validation_data=(X_val, y_val), callbacks=callbacks, verbose=1)

    # 评估模型
    y_val_true, y_val_pred, val_metrics = evaluate_model(best_model, X_val, y_val, scaler_y, "验证")
    y_test_true, y_test_pred, test_metrics = evaluate_model(best_model, X_test, y_test, scaler_y, "测试")

    # 计算残差
    residuals = y_test_true - y_test_pred

    # 可视化结果
    if not use_tuner:
        plot_results(history, y_val_true, y_val_pred, y_test_true, y_test_pred, residuals)

    # 交叉验证
    print("\n执行时间序列交叉验证...")
    cross_validate_model(file_path, window_size, n_splits)


# 运行主函数
if __name__ == "__main__":
    main()





# 加了热力图
# import math
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.stats import norm, probplot
# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import TimeSeriesSplit
# from keras.models import Sequential, Model
# from keras.layers import Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout, BatchNormalization
# from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# from keras.optimizers import Adam
# from keras import regularizers
# import matplotlib
# import warnings
# import keras_tuner as kt
#
# # 设置 Matplotlib 字体和后端
# try:
#     matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 优先使用 SimHei 字体
# except:
#     matplotlib.rcParams['font.sans-serif'] = ['Arial']  # 后备字体
# matplotlib.rcParams['axes.unicode_minus'] = False
# matplotlib.use('Agg')  # 使用非交互式后端
#
# # 忽略警告
# warnings.filterwarnings('ignore')
#
#
# # 数据预处理函数
# def preprocess_data(file_path, window_size=24, train_ratio=0.8, val_ratio=0.1):
#     """
#     预处理时间序列数据，包括加载、分割、窗口化和归一化。
#     返回训练、验证、测试数据和归一化器。
#     """
#     df = pd.read_csv(file_path)
#     if df.isna().sum().sum() > 0:
#         raise ValueError("数据中存在缺失值，请检查！")
#     if not np.all(df.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
#         raise ValueError("数据中存在非数值类型，请检查！")
#
#     data = df.values
#     print("数据形状:", data.shape)
#     print("目标值统计 - 均值: {:.4f}, 标准差: {:.4f}, 最小值: {:.4f}, 最大值: {:.4f}".format(
#         np.mean(data[:, -1]), np.std(data[:, -1]), np.min(data[:, -1]), np.max(data[:, -1])))
#
#     # 数据分割
#     total_size = len(data)
#     train_size = int(total_size * train_ratio)
#     val_size = int(total_size * val_ratio)
#     test_size = total_size - train_size - val_size
#
#     train_data = data[:train_size]
#     val_data = data[train_size:train_size + val_size]
#     test_data = data[train_size + val_size:]
#
#     # 窗口化
#     X_train, y_train = train_data[:, :window_size], train_data[:, -1]
#     X_val, y_val = val_data[:, :window_size], val_data[:, -1]
#     X_test, y_test = test_data[:, :window_size], test_data[:, -1]
#
#     # 归一化
#     scaler_y = StandardScaler()
#     y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
#     y_val = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
#     y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
#
#     scaler_x = StandardScaler()
#     X_train = scaler_x.fit_transform(X_train).reshape(X_train.shape[0], X_train.shape[1], 1)
#     X_val = scaler_x.transform(X_val).reshape(X_val.shape[0], X_val.shape[1], 1)
#     X_test = scaler_x.transform(X_test).reshape(X_test.shape[0], X_test.shape[1], 1)
#
#     return X_train, y_train, X_val, y_val, X_test, y_test, scaler_x, scaler_y
#
#
# # 构建模型函数
# def build_cnn_bilstm_model(window_size, l2_reg=0.001, dropout_rate=0.2, filters_1=64, filters_2=32, lstm_units_1=64,
#                            lstm_units_2=32):
#     """
#     构建 CNN-BiLSTM 模型，优化激活函数和丢弃率。
#     """
#     model = Sequential(name='cnn_bilstm')
#     model.add(Conv1D(filters=filters_1, kernel_size=5, activation='relu', input_shape=(window_size, 1),
#                      kernel_regularizer=regularizers.l2(l2_reg), name='conv1'))
#     model.add(BatchNormalization(name='bn1'))
#     model.add(MaxPooling1D(pool_size=2, name='pool1'))
#     model.add(Dropout(dropout_rate, name='dropout1'))
#
#     model.add(Conv1D(filters=filters_2, kernel_size=3, activation='relu',
#                      kernel_regularizer=regularizers.l2(l2_reg), name='conv2'))
#     model.add(BatchNormalization(name='bn2'))
#     model.add(MaxPooling1D(pool_size=2, name='pool2'))
#     model.add(Dropout(dropout_rate, name='dropout2'))
#
#     model.add(Bidirectional(LSTM(lstm_units_1, activation='tanh', return_sequences=True,
#                                  kernel_regularizer=regularizers.l2(l2_reg), name='bilstm1')))
#     model.add(Dropout(dropout_rate, name='dropout3'))
#     model.add(Bidirectional(LSTM(lstm_units_2, activation='tanh',
#                                  kernel_regularizer=regularizers.l2(l2_reg), name='bilstm2')))
#
#     model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_reg), name='dense1'))
#     model.add(Dropout(dropout_rate, name='dropout4'))
#     model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_reg), name='dense2'))
#     model.add(Dense(1, name='output'))
#
#     return model
#
#
# # 超参数调优模型
# def build_tuner_model(hp):
#     """
#     为 keras-tuner 构建模型，搜索超参数。
#     """
#     model = Sequential()
#     model.add(Conv1D(
#         filters=hp.Int('filters_1', 32, 128, step=32),
#         kernel_size=hp.Choice('kernel_size_1', [3, 5]),
#         activation='relu',
#         input_shape=(hp.Int('window_size', 12, 48, step=12), 1),
#         kernel_regularizer=regularizers.l2(hp.Float('l2', 1e-4, 1e-2, sampling='log')),
#         name='conv1'
#     ))
#     model.add(BatchNormalization(name='bn1'))
#     model.add(MaxPooling1D(pool_size=2, name='pool1'))
#     model.add(Dropout(hp.Float('dropout', 0.1, 0.3, step=0.1), name='dropout1'))
#
#     model.add(Conv1D(
#         filters=hp.Int('filters_2', 16, 64, step=16),
#         kernel_size=hp.Choice('kernel_size_2', [3, 5]),
#         activation='relu',
#         kernel_regularizer=regularizers.l2(hp.Float('l2', 1e-4, 1e-2, sampling='log')),
#         name='conv2'
#     ))
#     model.add(BatchNormalization(name='bn2'))
#     model.add(MaxPooling1D(pool_size=2, name='pool2'))
#     model.add(Dropout(hp.Float('dropout', 0.1, 0.3, step=0.1), name='dropout2'))
#
#     model.add(Bidirectional(LSTM(
#         units=hp.Int('lstm_units_1', 32, 128, step=32),
#         activation='tanh',
#         return_sequences=True,
#         kernel_regularizer=regularizers.l2(hp.Float('l2', 1e-4, 1e-2, sampling='log')),
#         name='bilstm1'
#     )))
#     model.add(Dropout(hp.Float('dropout', 0.1, 0.3, step=0.1), name='dropout3'))
#     model.add(Bidirectional(LSTM(
#         units=hp.Int('lstm_units_2', 16, 64, step=16),
#         activation='tanh',
#         kernel_regularizer=regularizers.l2(hp.Float('l2', 1e-4, 1e-2, sampling='log')),
#         name='bilstm2'
#     )))
#
#     model.add(Dense(
#         units=hp.Int('dense_units', 32, 128, step=32),
#         activation='relu',
#         kernel_regularizer=regularizers.l2(hp.Float('l2', 1e-4, 1e-2, sampling='log')),
#         name='dense1'
#     ))
#     model.add(Dropout(hp.Float('dropout', 0.1, 0.3, step=0.1), name='dropout4'))
#     model.add(Dense(1, name='output'))
#
#     model.compile(optimizer=Adam(hp.Float('lr', 1e-4, 1e-2, sampling='log')), loss='huber')
#     return model
#
#
# # 评估模型函数
# def evaluate_model(model, X, y, scaler_y, dataset_name="Validation"):
#     """
#     评估模型性能，计算 MSE、RMSE、MAE、MAPE 和 R²。
#     """
#     y_pred = model.predict(X, verbose=0)
#     y_pred = scaler_y.inverse_transform(y_pred).flatten()
#     y_true = scaler_y.inverse_transform(y.reshape(-1, 1)).flatten()
#
#     mse = mean_squared_error(y_true, y_pred)
#     rmse = math.sqrt(mse)
#     mae = mean_absolute_error(y_true, y_pred)
#     mape = mean_absolute_percentage_error(y_true, y_pred)
#     r2 = r2_score(y_true, y_pred)
#
#     print(f"\n{dataset_name} 集评估指标:")
#     print(f"MSE: {mse:.8f}")
#     print(f"RMSE: {rmse:.4f}")
#     print(f"MAE: {mae:.4f}")
#     print(f"MAPE: {mape:.4f}")
#     print(f"R²: {r2:.4f}")
#
#     return y_true, y_pred, {"mse": mse, "rmse": rmse, "mae": mae, "mape": mape, "r2": r2}
#
#
# # 可视化函数（新增 CNN 特征可视化）
# def plot_results(history, y_val_true, y_val_pred, y_test_true, y_test_pred, residuals, model, X_test,
#                  layer_name='conv1'):
#     """
#     绘制训练损失、预测结果、残差分析和 CNN 特征。
#     """
#     # 训练和验证损失
#     plt.figure(figsize=(12, 6))
#     plt.plot(history.history['loss'], label='训练损失', color='#00CED1')
#     plt.plot(history.history['val_loss'], label='验证损失', color='#8A2BE2')
#     plt.title('训练和验证损失')
#     plt.xlabel('Epochs')
#     plt.ylabel('损失')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('loss_plot.png')
#     plt.close()
#
#     # 验证集预测
#     plt.figure(figsize=(12, 6))
#     plt.plot(y_val_true, label='实际值', color='blue')
#     plt.plot(y_val_pred, label='预测值', color='red')
#     plt.title('验证集：实际值 vs 预测值')
#     plt.xlabel('时间步')
#     plt.ylabel('值')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('val_pred_plot.png')
#     plt.close()
#
#     # 测试集预测
#     plt.figure(figsize=(12, 6))
#     plt.plot(y_test_true, label='实际值', color='blue')
#     plt.plot(y_test_pred, label='预测值', color='red')
#     plt.title('测试集：实际值 vs 预测值')
#     plt.xlabel('时间步')
#     plt.ylabel('值')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('test_pred_plot.png')
#     plt.close()
#
#     # 残差图
#     plt.figure(figsize=(12, 6))
#     plt.plot(residuals, label='残差', color='blue')
#     plt.title('测试集残差图')
#     plt.xlabel('时间步')
#     plt.ylabel('残差')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('residuals_plot.png')
#     plt.close()
#
#     # 残差直方图和正态分布
#     plt.figure(figsize=(12, 6))
#     n, bins, patches = plt.hist(residuals, bins=50, density=True, alpha=0.6, color='blue', edgecolor='black')
#     mu, std = norm.fit(residuals)
#     x = np.linspace(min(residuals), max(residuals), 100)
#     p = norm.pdf(x, mu, std)
#     plt.plot(x, p, 'r-', linewidth=2, label='正态分布拟合')
#     plt.title('测试集残差直方图和正态分布拟合')
#     plt.xlabel('残差')
#     plt.ylabel('密度')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('residuals_hist_plot.png')
#     plt.close()
#
#     # 残差 Q-Q 图
#     plt.figure(figsize=(12, 6))
#     probplot(residuals, dist="norm", plot=plt)
#     plt.title('测试集残差 Q-Q 图')
#     plt.savefig('residuals_qq_plot.png')
#     plt.close()
#
#     # CNN 特征可视化
#     # 提取指定层的输出
#     feature_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
#     features = feature_model.predict(X_test[:5], verbose=0)  # 取前5个样本
#     print(f"{layer_name} 层特征形状: {features.shape}")
#
#     # 时序图：每个过滤器的激活值
#     n_filters = features.shape[2]
#     plt.figure(figsize=(12, 8))
#     for i in range(min(n_filters, 8)):  # 限制最多显示8个过滤器
#         plt.subplot(2, 4, i + 1)
#         plt.plot(features[0, :, i], label=f'过滤器 {i + 1}')
#         plt.title(f'过滤器 {i + 1}')
#         plt.xlabel('时间步')
#         plt.ylabel('激活值')
#         plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(f'{layer_name}_feature_timeseries.png')
#     plt.close()
#
#     # 热图：所有过滤器的激活值
#     plt.figure(figsize=(12, 6))
#     plt.imshow(features[0].T, aspect='auto', cmap='viridis', interpolation='nearest')
#     plt.colorbar(label='激活值')
#     plt.title(f'{layer_name} 层特征热图（样本 1）')
#     plt.xlabel('时间步')
#     plt.ylabel('过滤器')
#     plt.savefig(f'{layer_name}_feature_heatmap.png')
#     plt.close()
#
#
# # 时间序列交叉验证
# def cross_validate_model(file_path, window_size=24, n_splits=5):
#     """
#     使用 TimeSeriesSplit 进行交叉验证。
#     """
#     df = pd.read_csv(file_path)
#     data = df.values
#     tscv = TimeSeriesSplit(n_splits=n_splits)
#     metrics = []
#
#     for fold, (train_idx, val_idx) in enumerate(tscv.split(data), 1):
#         print(f"\n交叉验证折 {fold}")
#         train_data = data[train_idx]
#         val_data = data[val_idx]
#         X_train, y_train = train_data[:, :window_size], train_data[:, -1]
#         X_val, y_val = val_data[:, :window_size], val_data[:, -1]
#
#         scaler_y = StandardScaler()
#         y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
#         y_val = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
#
#         scaler_x = StandardScaler()
#         X_train = scaler_x.fit_transform(X_train).reshape(X_train.shape[0], X_train.shape[1], 1)
#         X_val = scaler_x.transform(X_val).reshape(X_val.shape[0], X_val.shape[1], 1)
#
#         model = build_cnn_bilstm_model(window_size)
#         model.compile(optimizer=Adam(learning_rate=0.001), loss='huber')
#
#         callbacks = [
#             EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
#             ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
#             ModelCheckpoint(f'best_model_fold_{fold}.keras', monitor='val_loss', save_best_only=True)
#         ]
#
#         model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val),
#                   callbacks=callbacks, verbose=1)
#
#         _, _, val_metrics = evaluate_model(model, X_val, y_val, scaler_y, f"交叉验证折 {fold}")
#         metrics.append(val_metrics)
#
#     avg_metrics = {key: np.mean([m[key] for m in metrics]) for key in metrics[0]}
#     print("\n平均交叉验证指标:", avg_metrics)
#     return avg_metrics
#
#
# # 主函数
# def main():
#     # 参数配置
#     file_path = "extracted_columns.csv"
#     window_size = 24
#     train_ratio = 0.8
#     val_ratio = 0.1
#     epochs = 200
#     batch_size = 32
#     n_splits = 5
#     use_tuner = False  # 是否使用超参数调优
#     feature_layer = 'conv1'  # 要可视化的 CNN 层
#
#     # 数据预处理
#     X_train, y_train, X_val, y_val, X_test, y_test, scaler_x, scaler_y = preprocess_data(
#         file_path, window_size, train_ratio, val_ratio)
#
#     if use_tuner:
#         # 超参数调优
#         tuner = kt.Hyperband(
#             build_tuner_model,
#             objective='val_loss',
#             max_epochs=100,
#             factor=3,
#             directory='tuner_dir',
#             project_name='cnn_bilstm_tuning'
#         )
#         tuner.search(X_train, y_train, epochs=50, validation_data=(X_val, y_val),
#                      callbacks=[EarlyStopping(monitor='val_loss', patience=10)])
#         best_model = tuner.get_best_models(num_models=1)[0]
#         print("最佳超参数:", tuner.get_best_hyperparameters()[0].values)
#     else:
#         # 使用默认模型
#         best_model = build_cnn_bilstm_model(window_size)
#         best_model.compile(optimizer=Adam(learning_rate=0.001), loss='huber')
#
#         callbacks = [
#             EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
#             ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
#             ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
#         ]
#
#         history = best_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
#                                  validation_data=(X_val, y_val), callbacks=callbacks, verbose=1)
#
#     # 评估模型
#     y_val_true, y_val_pred, val_metrics = evaluate_model(best_model, X_val, y_val, scaler_y, "验证")
#     y_test_true, y_test_pred, test_metrics = evaluate_model(best_model, X_test, y_test, scaler_y, "测试")
#
#     # 计算残差
#     residuals = y_test_true - y_test_pred
#
#     # 可视化结果
#     if not use_tuner:
#         plot_results(history, y_val_true, y_val_pred, y_test_true, y_test_pred, residuals, best_model, X_test,
#                      feature_layer)
#
#     # 交叉验证
#     print("\n执行时间序列交叉验证...")
#     cross_validate_model(file_path, window_size, n_splits)
#
#
# # 运行主函数
# if __name__ == "__main__":
#     main()