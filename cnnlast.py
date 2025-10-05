# Grok优化
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Bidirectional, LSTM, Conv1D, MaxPooling1D, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, r2_score
from keras.regularizers import l2
from math import sqrt
#
# # Load the CSV file (assuming the data is saved as a CSV)
# df = pd.read_csv("L:\English artical\Supplementary materials\extracted_feature.csv")
# TARGET_COLUMN = 'Y'
# TIME_STEPS = 17  # Increased to capture more temporal context
# BATCH_SIZE = 64  # Reduced for finer updates
# EPOCHS = 500
# VALIDATION_SPLIT = 0.20
#
# # Outlier removal (optional, adjust threshold as needed)
# df = df[df[TARGET_COLUMN].between(df[TARGET_COLUMN].quantile(0.01), df[TARGET_COLUMN].quantile(0.99))]
#
# # Feature selection with PCA (optional, reduce to top 100 components)
# X = df.drop(columns=[TARGET_COLUMN])
# pca = PCA(n_components=100)  # Adjust n_components based on explained variance
# X_reduced = pca.fit_transform(X)
# Y = df[TARGET_COLUMN]
#
# # Normalize the data
# scaler_X = MinMaxScaler(feature_range=(0, 1))
# scaler_Y = MinMaxScaler(feature_range=(0, 1))
# X_scaled = scaler_X.fit_transform(X_reduced)  # Use PCA-reduced features
# Y_scaled = scaler_Y.fit_transform(Y.values.reshape(-1, 1))
#
# # Combine X and Y for time-series slicing
# data = np.hstack((X_scaled, Y_scaled))
#
# # Time series data preparation
# def create_dataset(data, time_steps):
#     result = []
#     for i in range(len(data) - time_steps):
#         result.append(data[i:i + time_steps])
#     return np.array(result)
#
# # Prepare the dataset
# dataset = create_dataset(data, TIME_STEPS)
#
# # Split data into training and testing sets (80% training)
# train_size = int(0.8 * len(dataset))
# train, test = dataset[:train_size], dataset[train_size:]
#
# # Split into input (X) and output (Y)
# x_train = train[:, :-1, :-1]
# y_train = train[:, -1, -1]
# x_test = test[:, :-1, :-1]
# y_test = test[:, -1, -1]
#
# # Reshape X for input
# x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2])
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2])
#
# # Define the optimized model
# def build_model(input_shape):
#     model = Sequential()
#     # CNN Layers
#     model.add(Conv1D(filters=128, kernel_size=3, activation='relu', padding='same',
#                      kernel_regularizer=l2(0.01), input_shape=input_shape))
#     model.add(BatchNormalization())
#     model.add(MaxPooling1D(pool_size=2))
#     model.add(Dropout(0.4))  # Increased dropout to reduce overfitting
#
#     model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same',
#                      kernel_regularizer=l2(0.01)))
#     model.add(BatchNormalization())
#     model.add(MaxPooling1D(pool_size=2))
#     model.add(Dropout(0.4))
#
#     # BiLSTM Layers
#     model.add(Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.01))))
#     model.add(Dropout(0.4))
#     model.add(Bidirectional(LSTM(32, kernel_regularizer=l2(0.01))))
#     model.add(Dropout(0.4))
#
#     # Dense Layers
#     model.add(Dense(50, activation='relu', kernel_regularizer=l2(0.001)))
#     model.add(Dense(1, activation='linear'))  # Output layer for regression
#
#     model.compile(loss='mse', optimizer='adam', metrics=['mae'])
#     return model
#
# # Build and compile the model
# model = build_model((x_train.shape[1], x_train.shape[2]))
# print(model.summary())
#
# # Callbacks
# early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
#
# # Train the model
# history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
#                     validation_split=VALIDATION_SPLIT, verbose=1,
#                     callbacks=[early_stopping, reduce_lr])
#
# # Plot training and validation loss
# def plot_loss(history):
#     plt.figure(figsize=(10, 5))
#
#     # Extract training and validation loss
#     train_loss = history.history['loss']
#     val_loss = history.history['val_loss']
#     epochs = range(1, len(train_loss) + 1)
#
#     # Plot the original curves
#     plt.plot(epochs, train_loss, label='Training Loss', color='blue')
#     plt.plot(epochs, val_loss, label='Validation Loss', color='red')
#
#     # Dynamic buffer zone (e.g., 10% of the loss value at each epoch)
#     buffer_train = [tl * 0.1 for tl in train_loss]  # 10% buffer
#     buffer_val = [vl * 0.1 for vl in val_loss]  # 10% buffer
#
#     plt.fill_between(epochs, [max(0, tl - bt) for tl, bt in zip(train_loss, buffer_train)],
#                      [tl + bt for tl, bt in zip(train_loss, buffer_train)],
#                      color='blue', alpha=0.2)
#     plt.fill_between(epochs, [max(0, vl - bv) for vl, bv in zip(val_loss, buffer_val)],
#                      [vl + bv for vl, bv in zip(val_loss, buffer_val)],
#                      color='red', alpha=0.2)
#
#     # Customize the plot
#     plt.title('Training and Validation Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid()
#     plt.show()
#
#
# # Call the function with your history
# plot_loss(history)
#
# # Evaluate the model
# train_score = model.evaluate(x_train, y_train, verbose=0)
# test_score = model.evaluate(x_test, y_test, verbose=0)
#
# # Predictions
# y_pred = model.predict(x_test)
#
# # Plot predictions vs actual values
# def plot_predictions(y_test, y_pred):
#     plt.figure(figsize=(10, 5))
#     plt.plot(y_test, label='Actual', color='blue')
#     plt.plot(y_pred, label='Prediction', color='red')
#     plt.title('Predictions vs Actual Values')
#     plt.xlabel('Time Steps')
#     plt.ylabel('Scaled Y Values')
#     plt.legend()
#     plt.grid()
#     plt.show()
#
# plot_predictions(y_test, y_pred)
#
# # Reverse scaling
# y_test_original = scaler_Y.inverse_transform(y_test.reshape(-1, 1))
# y_pred_original = scaler_Y.inverse_transform(y_pred)
#
# # Calculate and print evaluation metrics
# mae = mean_absolute_error(y_test_original, y_pred_original)
# mape = np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100
# r2 = r2_score(y_test_original, y_pred_original)
#
# print('Train Score RMSE:', sqrt(train_score[0]))
# print('Test Score RMSE:', sqrt(test_score[0]))
# print('Train MAE:', train_score[1])
# print('Test MAE:', test_score[1])
# print('Mean Absolute Error:', mae)
# print('Mean Absolute Percentage Error:', mape)
# print('R² Score:', r2)
#
#
#
#

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ========= 1. 读取数据 =========
data = pd.read_csv("L:\English artical\CNN-BILISTM-main\paperdata.csv", header=None)

X = data.iloc[:, :-1].values  # 特征
y = data.iloc[:, -1].values   # 目标变量

# ========= 2. 数据预处理 =========
scaler = StandardScaler()
X = scaler.fit_transform(X)

# reshape -> (样本数, 波段数, 通道数)
X = X[..., np.newaxis]

# 划分训练 / 验证 / 测试集 (8:1:1)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# ========= 3. 搭建增强版 CNN-only 模型 =========
model = Sequential([
    Conv1D(filters=32, kernel_size=5, activation='relu',
           kernel_regularizer=l2(1e-4), padding='same',
           input_shape=(X.shape[1], 1)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Conv1D(filters=64, kernel_size=5, activation='relu',
           kernel_regularizer=l2(1e-4), padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Conv1D(filters=128, kernel_size=3, activation='relu',
           kernel_regularizer=l2(1e-4), padding='same'),
    BatchNormalization(),
    Dropout(0.4),

    Conv1D(filters=256, kernel_size=3, activation='relu',
           kernel_regularizer=l2(1e-4), padding='same'),
    BatchNormalization(),
    Dropout(0.4),

    Flatten(),
    Dense(256, activation='relu', kernel_regularizer=l2(1e-4)),
    Dropout(0.5),
    Dense(1)  # 回归输出
])


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='mse',
              metrics=['mae'])

# ========= 4. 回调函数 =========
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# ========= 5. 训练模型 =========
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    verbose=1,
    callbacks=[early_stop, reduce_lr]
)

# ========= 6. 测试集评估 =========
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test MSE: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import numpy as np

# ========= 模型预测 =========
y_pred = model.predict(X_test).flatten()

# ========= 评估指标 =========
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print("回归评估指标：")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R²   : {r2:.4f}")
print(f"MAPE : {mape:.4f}")
