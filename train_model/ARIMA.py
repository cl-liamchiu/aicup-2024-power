import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split

# 假設 data_ts 是 1098 rows × 120 columns 的資料
# 讀取數據
data = pd.read_csv('../data/preprocess_data/PowerData.csv')

# 1. 去除不必要的欄位，僅保留時間序列數據
data_ts = data.drop(columns=['LocationCode', 'Date_Day1', 'Date_Day2'])

# 2. 資料集切分
train_data, val_data = train_test_split(data_ts, test_size=0.2, random_state=42)

# 查看切分後的數據大小
print(f"Training set size: {train_data.shape}")
print(f"Validation set size: {val_data.shape}")

# 初始化列表來存儲結果
all_predictions = []
all_actuals = []
all_mses = []

# 3. 訓練每一行數據的 ARIMA 模型
for index, row in val_data.iterrows():
    try:
        # 提取前 72 個 time step 作為訓練數據
        train_series = row[:72].values
        
        # 使用 ARIMA 模型進行訓練
        model = ARIMA(train_series, order=(1, 1, 1))
        model_fit = model.fit()
        
        # 預測後 48 個 time step
        predictions = model_fit.forecast(steps=48)
        
        # 提取後 48 個 time step 作為真實值
        actual_values = row[72:].values
        
        # 計算 MSE
        mse = mean_squared_error(actual_values, predictions)
        
        # 存儲結果
        all_predictions.append(predictions)
        all_actuals.append(actual_values)
        all_mses.append(mse)
    except Exception as e:
        print(f"ARIMA model failed for row {index}: {e}")

# 計算平均 MSE
average_mse = np.mean(all_mses)
print(f"Validation set average MSE: {average_mse}")

# 4. 繪製第一行的結果
sample_index = 0  # 第一行的索引
plt.figure(figsize=(12, 6))
plt.plot(all_actuals[sample_index], label='Actual', marker='o')
plt.plot(all_predictions[sample_index], label='Predicted', marker='x')
plt.title('ARIMA')  # 設置標題
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.savefig('ARIMA_prediction.png', dpi=300)  # 保存圖表
plt.show()
