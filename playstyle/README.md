# 2023 AICUP GO

## 運行環境

### 使用python 3.10.5 64-bit (Window 10)
- 請自行安裝cuda
- 安裝tensorflow環境
```bash
pip install tensorflow==2.10.0
pip install tensorflow-addons==0.20.0
pip install keras==2.10.0
```

## 資料

- `CSVS`：存放訓練和測試資料集
- `result`：用於存放預測完的結果
- `saves`：存放訓練完的權重與模型
- `caches`: 若有使用caches就會存在這裡

- `remove_extra_commas.py`: 用於移除play_style_test_public和play_style_test_private中的多餘的逗號

# 程式碼介紹

## model

- `model.py`：用於呼叫train要使用的model

## preprocess

- `data_loader_PlayStyle.py`：包含了資料擴增、將資料轉成模型input
- `go_game.py`: 包含提子和氣的演算法

## 訓練

- train_play_style.py中設定好相關參數後直接執行即可

```bash
python ./train_play_style.py
```
### 可設定參數
- save_dir: 設定儲存權重的資料夾
- weight_name: 設定權重的名稱
- train_val_split: 設定驗證集的比例
- n_models: 設定要訓練幾個模型，每個模型的差異在於其train_dataset會有所不同
- epochs
- batch_size
- lr: 學習率
- n_last_move: 設定input要取到前幾手
- input_type':' input類型，有'with_liberty'和'no_liberty' 2種

### save的權重
```
---saves
     |--play_style
```
- xxx_model.h5: 訓練的模型
- best_xxx_acc.h5：validation 最高的accuracy
- best_xxx_loss.h5: validation 最低的loss

## 預測

- 在train.py中設定好相關參數後直接執行即可

```bash
python ./predict.py
```
### 可設定參數
- save_dir: 儲存權重的資料夾
- weight_name: 權重的名稱
- result_dir: 設定儲存預測結果的資料夾
- n_models: 設定有多少模型(權重)要預測
- batch_size
- n_last_move: 設定input要取到前幾手
- input_type: input類型，有'with_liberty'和'no_liberty' 2種
- n_tta:設定TTA的次數