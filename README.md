# AI CUP 2024 秋季賽 根據區域微氣候資料預測發電量競賽

隊伍：TEAM_6011
Private leaderboard：507376.3 / Rank 11

## 環境建置
使用 conda 建立環境，python 版本為 3.9.20
```bash
conda create -n ai_cup python=3.9.20
```

安裝相依套件
```bash
pip install -r requirements.txt
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118
```

## 資料集和模型下載
請至 https://drive.google.com/file/d/1zKq21xbNYOUwx7CXPpDrUjYUwjQkdkmy/view?usp=sharing 下載資料集和模型，並將其解壓縮至專案根目錄。

或者利用 gdown 下載後解壓縮
```bash
pip install gdown
gdown https://drive.google.com/uc?id=1zKq21xbNYOUwx7CXPpDrUjYUwjQkdkmy
unzip data_and_model.zip
```

## 資料夾說明
- `data/`：放置資料集，含原始資料集、外部資料以及預處理後的資料集
- `data_analysis/`：放置資料分析以及繪圖的程式碼
- `data_process/`：放置資料前處理的程式碼
- `figures/`：放置繪圖結果
- `result/`：放置預測結果以及繳交的檔案
- `train_model/`：放置訓練模型的程式碼，包含 ARIMA、LSTM、CatBoost，LSTM、CatBoost 包含在 main.py 中。

## 最佳結果複現
執行以下指令以進行模型的訓練與測試，並生成預測結果：

```bash
cd train_model
# 訓練模型
python main.py --stage=train --is72250=True
python main.py --stage=train
# 測試模型
python main.py --stage=test --is72250=True
python main.py --stage=test
```

上述步驟將在 `result/` 產生兩個預測結果檔案：`result_72250.csv` 和 `result.csv`。
由於 `result_72250.csv` 的內容有所缺失，需結合 `result.csv` 的結果來進行補充。

接著，前往 `data_process/` 資料夾，開啟並執行 `transform_data_to_answer.ipynb`（點擊 Run All 執行所有 cell）。請注意，執行前需將右上角的執行環境設置為前面建立的環境。

最後，轉換完成後的提交檔案將位於 `result/` 資料夾中，檔名為： `upload(answer)_predict_final_reshaped_result_72250_little96400_6timestep_nextfronthour.csv`。