import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os


def preprocess(data_path, variables_to_use, expected_X_columns, expected_y_columns, stage='train', train_scaler=None, y_scaled=True, is72250=False):
    """
    數據預處理函數，用於處理不同階段（train/val/test）的數據。

    參數:
        data_path (str): 數據文件的路徑。
        variables_to_use (dict): 包含欄位分類的字典，必須包含 `not_data`, `no_time_dependent`, 和 `time_dependent`。
        time_dependent_feature_count (int): 每個時間相關特徵的變數數量。
        n_time_for_X (int): 訓練階段的時間窗口大小。
        n_time_for_y (int): 輸出 y 的大小。
        stage (str): 當前階段，可為 'train', 'val', 或 'test'。

    返回:
        - train: 返回標準化的特徵以及 scaler (X_scaled, y_scaled, scaler_X, scaler_y)。
        - val: 返回標準化的特徵及目標值 (X_scaled, y_scaled, y)。
        - test: 返回標準化的特徵和非數據欄位 (X_scaled, not_data_df)。
    """
    # 讀取數據
    df = pd.read_csv(data_path)

    # 1. 篩選不進模型的欄位
    not_data_cols = variables_to_use.get('not_data', [])
    not_data_df = df[not_data_cols]

    # 2. 篩選非時間依賴的變數
    y_cols = variables_to_use.get('y_cols', [])
    if not y_cols:
        raise ValueError("y columns list is empty.")
    regex_pattern = "|".join(y_cols)
    y_df = df.filter(regex=regex_pattern)
    print(f"y columns: {y_df.columns.to_list()}")

    # 3. 篩選時間依賴的變數
    X_cols = variables_to_use.get('X_cols', [])
    if not X_cols:
        raise ValueError("X columns list is empty.")

    # 使用正則篩選時間相關欄位
    regex_pattern = "|".join(X_cols)
    X_df = df.filter(regex=regex_pattern)
    print(f"X columns: {X_df.columns.to_list()}")

    # 處理不同階段
    if stage == 'train':

        # 設置 X 和 y
        X = X_df
        y = y_df

        # 驗證欄位數量是否符合預期
        if X.shape[1] != expected_X_columns:
            raise ValueError(
                f"X column count mismatch: Expected {expected_X_columns}, got {X.shape[1]}")
        if y.shape[1] != expected_y_columns:
            raise ValueError(
                f"y column count mismatch: Expected {expected_y_columns}, got {y.shape[1]}")

        # 標準化
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)

        scaler_y = StandardScaler()
        if y_scaled:
            y_scaled = scaler_y.fit_transform(y)
        else:
            y_scaled = y.to_numpy()

        # 保存標準化器
        if is72250:
            scaler_filename = 'scaler_72250.pkl'
        else:
            scaler_filename = 'scaler.pkl'
        joblib.dump({'X': scaler_X, 'y': scaler_y}, scaler_filename)
        print(f"Scaler saved as '{scaler_filename}'")

        print(f"X shape: {X_scaled.shape}, y shape: {y_scaled.shape}")
        return X_scaled, y_scaled, scaler_X, scaler_y

    elif stage == 'val':

        # 設置 X 和 y
        X = X_df
        y = y_df

        # 驗證欄位數量是否符合預期
        if X.shape[1] != expected_X_columns:
            raise ValueError(
                f"X column count mismatch: Expected {expected_X_columns}, got {X.shape[1]}")
        if y.shape[1] != expected_y_columns:
            raise ValueError(
                f"y column count mismatch: Expected {expected_y_columns}, got {y.shape[1]}")

        # 取得標準化器
        if train_scaler is None:
            raise ValueError("train_scaler is required for validation stage.")

        scaler_X = train_scaler['X']
        scaler_y = train_scaler['y']

        # 標準化
        X_scaled = scaler_X.transform(X)

        if y_scaled:
            y_scaled = scaler_y.transform(y)
        else:
            y_scaled = y.to_numpy()

        print(f"X shape: {X_scaled.shape}, y shape: {y_scaled.shape}")
        return X_scaled, y_scaled, y

    elif stage == 'test':

        # 設置 X
        X = X_df

        # 驗證欄位數量是否符合預期
        if X.shape[1] != expected_X_columns:
            raise ValueError(
                f"X column count mismatch: Expected {expected_X_columns}, got {X.shape[1]}")

        # 標準化
        print("is72250: ", is72250)
        X_scaled = _scale_features(X, is72250)
    

        print(f"X shape: {X_scaled.shape}")
        return X_scaled, not_data_df
    else:
        raise ValueError("Invalid stage. Must be 'train', 'val', or 'test'.")


def _scale_features(X, is72250):
    """
    加載訓練時保存的標準化器，並對特徵進行標準化。
    """
    if is72250:
        print("Loading 72250 scaler")
        scaler = joblib.load('scaler_72250.pkl')['X']
    else:
        scaler = joblib.load('scaler.pkl')['X']
    return scaler.transform(X)


def postprocess(y_pred, not_data_df, y_scaled):
    # 將標準化的 y_pred 反標準化
    if y_scaled:
        y_pred = inverse_scale_output(y_pred)
    else:
        y_pred = y_pred

    # 根據 y_pred 欄位數動態生成欄位名稱
    # y_columns = [f"time_{i}_Power(mW)" for i in range(73, 121)]
    
    y_columns = ["Power_1", "Power_2", "Power_3", "Power_4", "Power_5", "Power_6"]
    y_pred_df = pd.DataFrame(y_pred, columns=y_columns)

    # 拼接結果
    result = pd.concat([not_data_df.reset_index(drop=True), y_pred_df], axis=1)

    return result


def inverse_scale_output(y_pred):
    scaler_filename = 'scaler.pkl'
    if not os.path.exists(scaler_filename):
        raise FileNotFoundError(
            f"Scaler file '{scaler_filename}' does not exist.")

    data = joblib.load(scaler_filename)
    scaler_y = data['y']

    return scaler_y.inverse_transform(y_pred)


if __name__ == '__main__':
    not_data = ["LocationCode", "Date_Day1", "Date_Day2"]
    no_time_dependent = []
    time_dependent = ["Pressure", "Temperature",
                      "Humidity", "Sunlight", "Power"]

    variables = {
        'not_data': not_data,
        'no_time_dependent': no_time_dependent,
        'time_dependent': time_dependent
    }

    preprocess('Data/train_data.csv', variables,
               len(time_dependent), stage='train')
