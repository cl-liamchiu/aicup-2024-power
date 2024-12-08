import argparse
import os
import joblib
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import numpy as np

from data_process import preprocess, postprocess, inverse_scale_output

from LSTM import MultiStepLSTM
from torch.utils.data import DataLoader, TensorDataset

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd


def train_catboost_single_with_residual(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, is72250=False):
    mse_list = []
    rmse_list = []
    r2_list = []
    cat_models = []
    cat_models_residual = []
    y_list = []

    for i, target in enumerate(['Power_1', 'Power_2', 'Power_3', 'Power_4', 'Power_5', 'Power_6']):
        print(f"正在訓練目標變數：{target}")

        cat_model = CatBoostRegressor(iterations=10000, learning_rate=0.05, depth=10,
                                      random_seed=42, verbose=100, early_stopping_rounds=50)
        cat_model_residual = CatBoostRegressor(iterations=10000, learning_rate=0.05, depth=10,
                                               random_seed=42, verbose=100, early_stopping_rounds=50)

        cat_model.fit(X_train_scaled, y_train_scaled[:, i], eval_set=(
            X_test_scaled, y_test_scaled[:, i]))

        train_y_pred = cat_model.predict(X_train_scaled)
        train_residual = y_train_scaled[:, i] - train_y_pred

        test_y_pred = cat_model.predict(X_test_scaled)
        test_residual = y_test_scaled[:, i] - test_y_pred

        cat_model_residual.fit(X_train_scaled, train_residual, eval_set=(
            X_test_scaled, test_residual))

        y_pred_cat = cat_model.predict(
            X_test_scaled) + cat_model_residual.predict(X_test_scaled)

        mse = mean_squared_error(y_test_scaled[:, i], y_pred_cat)
        rmse = mse ** 0.5
        r2 = r2_score(y_test_scaled[:, i], y_pred_cat)

        mse_list.append(mse)
        rmse_list.append(rmse)
        r2_list.append(r2)
        y_list.append(y_pred_cat)

        # 印出當前目標變數的結果
        print(f"{target} - 均方誤差 (MSE): {mse}")
        print(f"{target} - 均方根誤差 (RMSE): {rmse}")
        print(f"{target} - 決定係數 (R²): {r2}")

        cat_models.append(cat_model)
        cat_models_residual.append(cat_model_residual)

    # 印出所有目標變數的評估結果
    for i, target in enumerate(['Power_1', 'Power_2', 'Power_3', 'Power_4', 'Power_5', 'Power_6']):
        print(f"\n{target} 的最終結果：")
        print(f"MSE: {mse_list[i]}")
        print(f"RMSE: {rmse_list[i]}")
        print(f"R²: {r2_list[i]}")

    # 印出平均 MSE
    print("Average MSE:", np.mean(mse_list))

    # 儲存模型
    if is72250:
        output_filename = 'catboost_model_single_with_residual_72250.pkl'
    else:
        output_filename = 'catboost_model_single_with_residual.pkl'
    joblib.dump({'models': cat_models,
                'models_residual': cat_models_residual}, output_filename)
    print(f"Model saved as '{output_filename}'")

    y_list = np.array(y_list).T
    return y_list


def train_catboost_single(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled):
    mse_list = []
    rmse_list = []
    r2_list = []
    cat_models = []
    y_list = []
    print(y_train_scaled.shape)

    col = [f'Power_{i}' for i in range(1, 1+y_train_scaled.shape[1])]
    print(col)
    for i, target in enumerate(col):
        print(f"正在訓練目標變數：{target}")

        cat_model = CatBoostRegressor(iterations=10000, learning_rate=0.05, depth=10,
                                      random_seed=42,
                                      verbose=100, early_stopping_rounds=50)
        cat_model.fit(X_train_scaled, y_train_scaled[:, i], eval_set=(
            X_test_scaled, y_test_scaled[:, i]))

        y_pred_cat = cat_model.predict(X_test_scaled)

        mse = mean_squared_error(y_test_scaled[:, i], y_pred_cat)
        rmse = mse ** 0.5
        r2 = r2_score(y_test_scaled[:, i], y_pred_cat)

        mse_list.append(mse)
        rmse_list.append(rmse)
        r2_list.append(r2)
        y_list.append(y_pred_cat)

        # 印出當前目標變數的結果
        print(f"{target} - 均方誤差 (MSE): {mse}")
        print(f"{target} - 均方根誤差 (RMSE): {rmse}")
        print(f"{target} - 決定係數 (R²): {r2}")

        cat_models.append(cat_model)

    # 印出所有目標變數的評估結果
    for i, target in enumerate(col):
        print(f"\n{target} 的最終結果：")
        print(f"MSE: {mse_list[i]}")
        print(f"RMSE: {rmse_list[i]}")
        print(f"R²: {r2_list[i]}")

    # 印出平均 MSE
    print("Average MSE:", np.mean(mse_list))

    # 儲存模型
    output_filename = 'catboost_model_single.pkl'
    joblib.dump(
        {'models': cat_models}, output_filename)
    print(f"Model saved as '{output_filename}'")

    y_list = np.array(y_list).T

    return y_list


def train_catboost(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled):

    model = CatBoostRegressor(
        random_state=42,
        loss_function='MultiRMSE',
        task_type='GPU',
        devices='2',
        iterations=10000,
        learning_rate=0.05,
        depth=10,
        early_stopping_rounds=50,
        boosting_type='Plain',  # for multiRMSE
    )

    model.fit(
        X_train_scaled,
        y_train_scaled,
        eval_set=(X_test_scaled, y_test_scaled),
        verbose=100
    )

    y_pred_scaled = model.predict(X_test_scaled)
    print(f"Best iteration: {model.best_iteration_}")

    output_filename = 'catboost_model.pkl'
    joblib.dump({'model': model}, output_filename)
    print(f"Model saved as '{output_filename}'")

    return y_pred_scaled


def train_lstm(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled):
    num_layers = 1
    hidden_layer_size = 50
    learning_rate = 0.001
    batch_size = 32
    epochs = 3000
    validate_every = 1
    seed = 42
    n_time_steps = 72
    n_features = 1

    assert X_train_scaled.shape[1] == n_time_steps * \
        n_features, "Feature 數量與時間步數與變數數量不符"

    # (資料數量, 時間步數, 變數數量)
    X_train_scaled = X_train_scaled.reshape(-1, n_time_steps, n_features)
    # (資料數量, 時間步數, 變數數量)
    X_test_scaled = X_test_scaled.reshape(-1, n_time_steps, n_features)

    def set_random_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

    set_random_seed(seed)

    X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train = torch.tensor(y_train_scaled, dtype=torch.float32)
    X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test = torch.tensor(y_test_scaled, dtype=torch.float32)

    # 建立訓練與驗證資料集
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    # 資料加載器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    input_size = n_features  # features 數量
    output_size = y_train.shape[1]  # 預測的 y 的數量

    model = MultiStepLSTM(input_size=input_size, hidden_layer_size=hidden_layer_size,
                          output_size=output_size, num_layers=num_layers)

    # 設置損失函數和優化器
    criterion = nn.MSELoss()  # 適用於迴歸問題
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 將模型移至裝置 (如 GPU，如果可用)
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 用 loss 來選擇最佳模型
    best_loss = float('inf')

    training_losses = []
    validation_losses = []
    for epoch in range(epochs):
        model.train()  # 設置為訓練模式
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()  # 重設梯度
            predictions = model(X_batch)  # 前向傳播
            loss = criterion(predictions, y_batch)  # 計算損失
            loss.backward()  # 反向傳播
            optimizer.step()  # 更新參數

            train_loss += loss.item()

        # 平均損失
        train_loss /= len(train_loader)
        if (epoch + 1) % validate_every == 0:
            training_losses.append(train_loss)
            print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss:.4f}")

            model.eval()  # 設置為評估模式
            val_loss = 0.0

            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    predictions = model(X_batch)
                    loss = criterion(predictions, y_batch)
                    val_loss += loss.item()

            val_loss /= len(test_loader)
            validation_losses.append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_state = model.state_dict()
                print(
                    f"Validation Loss after Epoch {epoch+1}: {best_loss:.4f} (Model Updated)")
            else:
                print(f"Validation Loss after Epoch {epoch+1}: {val_loss:.4f}")

            model.train()

    # 預測
    best_model = MultiStepLSTM(input_size=input_size, hidden_layer_size=hidden_layer_size,
                               output_size=output_size, num_layers=num_layers)
    best_model.load_state_dict(best_model_state)

    y_pred = []
    with torch.no_grad():
        best_model.eval()
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            predictions = model(X_batch)
            y_pred.append(predictions.cpu().numpy())

    y_pred = np.concatenate(y_pred, axis=0)
    y_pred = y_pred.reshape(-1, output_size)

    # 保存模型
    output_filename = 'lstm_model.pth'
    if best_model_state is not None:
        torch.save(best_model_state, output_filename)
        print(
            f"Best model saved to {output_filename} with validation loss: {best_loss:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(training_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('lstm_loss_plot.png')

    return y_pred


def inverse_transform_power(value):
    return value * 2626.48 + 0.0


def evaluate_model(y_test, y_pred):
    # y_test = inverse_transform_power(y_test)
    # y_pred = inverse_transform_power(y_pred)

    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    return mse


def train_pipeline(train_data_path, val_data_path, variables, expected_X_columns, expected_y_columns, model_name, y_scaled, is72250=False):
    X_train_scaled, y_train_scaled, X_scaler, y_scaler = preprocess(
        train_data_path,
        variables,
        expected_X_columns,
        expected_y_columns,
        stage='train',
        y_scaled=y_scaled,
        is72250=is72250
    )

    train_scaler = {
        'X': X_scaler,
        'y': y_scaler
    }

    X_test_scaled, y_test_scaled, y_test = preprocess(
        val_data_path,
        variables,
        expected_X_columns,
        expected_y_columns,
        stage='val',
        train_scaler=train_scaler,
        y_scaled=y_scaled,
        is72250=is72250
    )

    print(model_name)

    if model_name == 'LSTM':
        y_pred_scaled = train_lstm(
            X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled)
    elif model_name == 'CatBoost':
        y_pred_scaled = train_catboost(
            X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled)
    elif model_name == 'CatBoost_single':
        y_pred_scaled = train_catboost_single(
            X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled)
    elif model_name == 'CatBoost_single_with_residual':
        y_pred_scaled = train_catboost_single_with_residual(
            X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, is72250=is72250)

    if y_scaled:
        y_pred = inverse_scale_output(y_pred_scaled.reshape(-1, 1))
    else:
        y_pred = y_pred_scaled
    mse = evaluate_model(y_test, y_pred)
    print(y_test.shape)
    print(y_pred.shape)

    # 繪製圖表
    if model_name != 'CatBoost_single':

        sample_index = 0
        row_numpy = y_test.iloc[sample_index].to_numpy()
        plt.figure(figsize=(12, 6))
        plt.plot(row_numpy, label='Actual', marker='o')
        plt.plot(y_pred[sample_index], label='Predicted', marker='x')
        plt.title(model_name)  # 設置標題
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()

        # 保存圖表到文件
        plt.savefig(f'{model_name}_prediction_plot.png', dpi=300)  # 保存為高解析度圖片
    return mse


def predict_pipeline(data_path, variables, model_filename, expected_X_columns, expected_y_columns, y_scaled, is72250):
    if not os.path.exists(model_filename):
        raise FileNotFoundError(
            f"Model file '{model_filename}' does not exist.")

    X_test_scaled, not_data_df = preprocess(
        data_path,
        variables,
        expected_X_columns,
        expected_y_columns,
        stage='test',
        y_scaled=y_scaled,
        is72250=is72250
    )
    model_data = joblib.load(model_filename)
    models = model_data['models']
    residual_models = model_data['models_residual']

    y_list = []
    for i, model in enumerate(models):
        y_pred = model.predict(X_test_scaled)
        y_pred_residual = residual_models[i].predict(X_test_scaled)
        y_pred = y_pred + y_pred_residual
        y_list.append(y_pred)

    y_pred_scaled = np.array(y_list).T

    result = postprocess(y_pred_scaled, not_data_df, y_scaled)

    if is72250:
        output_filename = '../result/result_72250.csv'
    else:
        output_filename = '../result/result.csv'
    result.to_csv(output_filename, index=False)
    print(f"Predictions saved as '{output_filename}'")
    return result


if __name__ == '__main__':
    # 使用 argparse 處理命令行參數
    parser = argparse.ArgumentParser(description="Specify the stage for the script.")
    parser.add_argument('--stage', type=str, choices=['train', 'test'], required=True,
                        help="Specify the stage: 'train' or 'test'")
    parser.add_argument('--is72250', type=bool, default=False,
                        help="Specify whether the data is 72250 data or not")
    
    # 解析參數
    args = parser.parse_args()
    
    # 根據輸入的 stage 進行對應操作
    stage = args.stage  # 解析到的參數值
    is72250 = args.is72250

    # 'LSTM', 'CatBoost', 'CatBoost_single', 'CatBoost_single_with_residual'
    model_name = 'CatBoost_single_with_residual'

    data_name = "data_after_1_6power"

    augmentation = 0 # 0: 不使用資料增強 1: 有高斯噪音生成 2: 用 vae 生成

    interpolation = False

    if data_name == "time_series_power_only":
        expected_X_columns = 72
        expected_y_columns = 48

        variables = {
            'not_data': ["LocationCode", "Date_Day1", "Date_Day2"],
            'y_cols': ["(0[9]:[0-5]0|1[0-6]:[0-5]0)_2_Power"],
            'X_cols': ["1_Power", "^0[7-8]:[0-5]0_2_Power"]
        }
        if stage == 'train':
            if augmentation==0:
                if interpolation:
                    train_data_path = '../data/data_augmentation/AllVariablesData_interpolation_train.csv'
                else:
                    train_data_path = '../data/data_augmentation/AllVariablesData_train.csv'
            elif augmentation==1:
                if interpolation:
                    train_data_path = '../data/data_augmentation/AllVariablesData_interpolation_train_augmentation.csv'
                else:
                    train_data_path = '../data/data_augmentation/AllVariablesData_train_augmentation.csv'
            elif augmentation==2:
                if interpolation:
                    train_data_path = '../data/data_augmentation_vae/AllVariablesData_interpolation_train_vae.csv'
                else:
                    train_data_path = '../data/data_augmentation/AllVariablesData_train_vae.csv'
            
            val_data_path = '../data/preprocess_data/AllVariablesData_val.csv'
            train_pipeline(train_data_path, val_data_path, variables,
                           expected_X_columns, expected_y_columns, model_name, y_scaled=False)
            
    elif data_name == "time_series_with_multiple_variables":
        expected_X_columns = 288
        expected_y_columns = 48

        variables = {
            'not_data': ["LocationCode", "Date_Day1", "Date_Day2"],
            'y_cols': ["(0[9]:[0-5]0|1[0-6]:[0-5]0)_2_Power"],
            'X_cols': ["1_Pressure", "1_Temperature", "1_Humidity", "1_Power",
                       "^0[7-8]:[0-5]0_2_Pressure", "^0[7-8]:[0-5]0_2_Temperature", "^0[7-8]:[0-5]0_2_Humidity", "^0[7-8]:[0-5]0_2_Power"]
        }
        if stage == 'train':
            if augmentation==0:
                if interpolation:
                    train_data_path = '../data/data_augmentation/AllVariablesData_interpolation_train.csv'
                else:
                    train_data_path = '../data/data_augmentation/AllVariablesData_train.csv'
            elif augmentation==1:
                if interpolation:
                    train_data_path = '../data/data_augmentation/AllVariablesData_interpolation_train_augmentation.csv'
                else:
                    train_data_path = '../data/data_augmentation/AllVariablesData_train_augmentation.csv'
            elif augmentation==2:
                if interpolation:
                    train_data_path = '../data/data_augmentation_vae/AllVariablesData_interpolation_train_vae.csv'
                else:
                    train_data_path = '../data/data_augmentation/AllVariablesData_train_vae.csv'
            
            val_data_path = '../data/preprocess_data/AllVariablesData_val.csv'
            train_pipeline(train_data_path, val_data_path, variables,
                           expected_X_columns, expected_y_columns, model_name, y_scaled=False)

    elif data_name == "data_after":
        is72250 = False
        if stage == 'train':
            if is72250:
                expected_X_columns = 59
                expected_y_columns = 1

                variables = {
                    'not_data': ["Date", "TimeMinGroup", "yyyymmddhh", "LocationCode"],
                    'y_cols': ["Power"],
                    'X_cols': ["^Time$", ".*466990$", ".*72250$",  "nearby", "Lat", "Lon", "Month", "Season", "direction", "floor"]
                }
                train_data_path = '../data/preprocess_data/oneday_weather/data_after_72250_train.csv'
                val_data_path = '../data/preprocess_data/oneday_weather/data_after_72250_val.csv'
            else:
                expected_X_columns = 41
                expected_y_columns = 1

                variables = {
                    'not_data': ["Date", "TimeMinGroup", "yyyymmddhh", "LocationCode"],
                    'y_cols': ["Power"],
                    'X_cols': ["^Time$", ".*466990$",  "nearby", "Lat", "Lon", "Month", "Season", "direction", "floor"]
                }
                train_data_path = '../data/preprocess_data/oneday_weather/data_after_train.csv'            
                val_data_path = '../data/preprocess_data/oneday_weather/data_after_val.csv'
            train_pipeline(train_data_path, val_data_path, variables,
                           expected_X_columns, expected_y_columns, model_name, y_scaled=False)
    
    elif data_name == "data_after_1":
        is72250 = True
        if stage == 'train':
            if is72250:
                expected_X_columns = 76
                expected_y_columns = 1

                variables = {
                    'not_data': ["Date", "TimeMinGroup", "yyyymmddhh"],
                    'y_cols': ["Power"],
                    'X_cols': ["LocationCode", "^Time$", ".*466990$", ".*72250$",  "nearby", "Lat", "Lon", "Month", "Season", "direction", "floor"]
                }
                train_data_path = '../data/preprocess_data/oneday_weather/data_after_72250_1_train.csv'
                val_data_path = '../data/preprocess_data/oneday_weather/data_after_72250_1_val.csv'
            else:
                expected_X_columns = 58
                expected_y_columns = 1

                variables = {
                    'not_data': ["Date", "TimeMinGroup", "yyyymmddhh"],
                    'y_cols': ["Power"],
                    'X_cols': ["LocationCode", "^Time$", ".*466990$",  "nearby", "Lat", "Lon", "Month", "Season", "direction", "floor"]
                }
                
                train_data_path = '../data/preprocess_data/oneday_weather/data_after_1_train.csv'            
                val_data_path = '../data/preprocess_data/oneday_weather/data_after_1_val.csv'
            train_pipeline(train_data_path, val_data_path, variables,
                           expected_X_columns, expected_y_columns, model_name, y_scaled=False)

    elif data_name == "data_after_1_6power":
        if stage == 'train':
            print(is72250)
            if is72250:
                expected_X_columns = 172
                expected_y_columns = 6

                location_codes = [
                    "LocationCode_1", "LocationCode_2", "LocationCode_3", "LocationCode_4",
                    "LocationCode_5", "LocationCode_6", "LocationCode_7", "LocationCode_8",
                    "LocationCode_9", "LocationCode_10", "LocationCode_11", "LocationCode_12",
                    "LocationCode_13", "LocationCode_14", "LocationCode_15", "LocationCode_16",
                    "LocationCode_17"
                ]

                variables = {
                    'not_data': ["Date", "Time"] + location_codes,  # for six_timestep
                    'y_cols': ["Power"],
                    'X_cols': ["^Time$", ".*72250",".*466990$", ".*466990_date_time_plus_1$", ".*466990_date_time_minus_1$", ".*466990_date_minus_1_time$", "nearby", "Lat", "Lon", "Month", "Season", "direction", "floor", "LocationCode"]
                }
                train_data_path = '../data/preprocess_data/oneday_weather_6timestep/data_after_72250_1_6power_train.csv'            
                val_data_path = '../data/preprocess_data/oneday_weather_6timestep/data_after_72250_1_6power_val.csv'

            else:
                expected_X_columns = 100
                expected_y_columns = 6

                location_codes = [
                    "LocationCode_1", "LocationCode_2", "LocationCode_3", "LocationCode_4",
                    "LocationCode_5", "LocationCode_6", "LocationCode_7", "LocationCode_8",
                    "LocationCode_9", "LocationCode_10", "LocationCode_11", "LocationCode_12",
                    "LocationCode_13", "LocationCode_14", "LocationCode_15", "LocationCode_16",
                    "LocationCode_17"
                ]

                variables = {
                    'not_data': ["Date", "Time"] + location_codes,  # for six_timestep
                    'y_cols': ["Power"],
                    'X_cols': ["^Time$", ".*466990$", ".*466990_date_time_plus_1$", ".*466990_date_time_minus_1$", ".*466990_date_minus_1_time$", "nearby", "Lat", "Lon", "Month", "Season", "direction", "floor", "LocationCode"]
                }
                
                train_data_path = '../data/preprocess_data/oneday_weather_6timestep/data_after_1_6power_train.csv'            
                val_data_path = '../data/preprocess_data/oneday_weather_6timestep/data_after_1_6power_val.csv'
            train_pipeline(train_data_path, val_data_path, variables,
                           expected_X_columns, expected_y_columns, model_name, y_scaled=False, is72250=is72250)
        
    if stage == 'test':
        print(is72250)
        if is72250:
            model_filename = 'catboost_model_single_with_residual_72250.pkl'
            expected_X_columns = 172
            expected_y_columns = 6

            location_codes = [
                "LocationCode_1", "LocationCode_2", "LocationCode_3", "LocationCode_4",
                "LocationCode_5", "LocationCode_6", "LocationCode_7", "LocationCode_8",
                "LocationCode_9", "LocationCode_10", "LocationCode_11", "LocationCode_12",
                "LocationCode_13", "LocationCode_14", "LocationCode_15", "LocationCode_16",
                "LocationCode_17"
            ]

            variables = {
                'not_data': ["Date", "Time"] + location_codes,  # for six_timestep
                'y_cols': ["Power"],
                'X_cols': ["^Time$", ".*72250",".*466990$", ".*466990_date_time_plus_1$", ".*466990_date_time_minus_1$", ".*466990_date_minus_1_time$", "nearby", "Lat", "Lon", "Month", "Season", "direction", "floor", "LocationCode"]
            }
            test_data_path = '../data/preprocess_data/oneday_weather_6timestep/data_after_72250_test_1_6power.csv'

        else:
            model_filename = f'catboost_model_single_with_residual.pkl'
            expected_X_columns = 100
            expected_y_columns = 6

            location_codes = [
                "LocationCode_1", "LocationCode_2", "LocationCode_3", "LocationCode_4",
                "LocationCode_5", "LocationCode_6", "LocationCode_7", "LocationCode_8",
                "LocationCode_9", "LocationCode_10", "LocationCode_11", "LocationCode_12",
                "LocationCode_13", "LocationCode_14", "LocationCode_15", "LocationCode_16",
                "LocationCode_17"
            ]

            variables = {
                'not_data': ["Date", "Time"] + location_codes,  # for six_timestep
                'y_cols': ["Power"],
                'X_cols': ["^Time$", ".*466990$", ".*466990_date_time_plus_1$", ".*466990_date_time_minus_1$", ".*466990_date_minus_1_time$", "nearby", "Lat", "Lon", "Month", "Season", "direction", "floor", "LocationCode"]
            }
            
            test_data_path = '../data/preprocess_data/oneday_weather_6timestep/data_after_test_1_6power.csv'
        
        predict_pipeline(test_data_path, variables, model_filename,
                         expected_X_columns, expected_y_columns, y_scaled=False, is72250=is72250)
