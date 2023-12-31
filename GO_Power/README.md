# 2023 AICUP GO

## 運行環境

### 使用python 3.10.5 64-bit (Window 10)
- 請自行安裝cuda
- 安裝tensorflow環境
```bash
pip install tensorflow==2.10.0
pip install keras==2.10.0
```

## 資料

- `CSVS`：存放訓練資料集
- `test_dataset`：存放測試資料集
- `result`：用於存放預測完的結果
- `save`：存放訓練完的權重

處理腳本

- `utils_/concate.py`: 將Public跟Private資料集合併
- `utils_/data_analyse.py`: 生成平衡過的.npy檔，用於訓練中平衡資料，同時也計算出大約的總資料量，用於train.py中的DATA_NUM

# 程式碼介紹

## model

- 用於存放model的資料夾
- `model/create_model.py`：用於呼叫train要使用的model

## preprocess

- 用於存放資料處裡的程式
- `preprocess/dataloader.py`：包含了資料擴增、輸入轉棋盤、平衡資料等。
- `preprocess/input_type.py`：為設計的input。

## utils_

- 包含一些工具
- `utils_/go_game.py`：包含提子跟氣的程式
- `utils_/lib.py`：包含訓練與預測的一些函式

## 訓練

- 在train.yaml中調整好參數
- 執行train.py即可

```bash
python ./train.py
```

### train.yaml之參數
- power：選擇訓練的資料集 `kyu/dan`
- input_data：input的格式與shape `baseline:19,19,4/inputs1:19,19,4/input_v2:19,19,4/time_input_v2:19,19,12/input_v3:19,19,6/time_input_v3:19,19,18/time_5_input_v4:19,19,25/time_7_input_v4:19,19,35/time_3_input_v4`
- model：選擇模型 `move_pred_mix_v2/kata_cnn_2/cnn_model`
- augment：選擇擴增模式。`org/1/8`org表示不進行隨機相位旋轉，1表示將原本資料隨機做相位旋轉，8表示將原本資料乘8倍，使用所有相位的資料進行訓練(總共8個相位)
- val_ratio：validation data切的倍率，0.1表示train比validation為9:1
- batch_size：batch大小 (A100使用2048, 3090使用1024)
- epochs：跑幾個epoch
- cache_path：放cache的資料夾
- shuffle_seed：shuffle的種子碼，都設定為1234
- train_status：`1/2` 1代表validation data會切在前面，2代表會切在後面
- weight_path：表示選擇的pretrain weight
- evaluate：代表不做訓練，而是測試當前weight的效能 (用validation做測試)

### save的權重
```
---save
     |--save_dan
     |--save_kyu
```
- dan的資料會存放在save_dan，kyu則會存在save_kyu
- 每個訓練完的資料會存成一個資料夾
- 資料夾名稱：{power}_{input_name}_{model}_{是否augment}_{train_status}_{i}
     - `tensorboard`： 儲存訓練中的結果使用tensorboard保存，可用指令 `tensorboard --logdir=tensorboard` 來監看
     - `config.yaml`： 預測使用的config
     - `model.keras`： 儲存的模型
     - `best_acc.hdf5`：儲存validation 最高的accuracy
     - `best_loss.hdf5`：儲存validation 最低的loss

## 預測

- 在predict.yaml中調整好參數
- 執行predict.py即可

```bash
python ./predict.py
```
### predict.yaml之參數

- kyu_predict： `True/False` 是否要進行Kyu的預測
- dan_predict： `True/False` 是否要進行Dan的預測
- csvs： 會將list中的csv都合併
- dan or kyu： 進行dan或是kyu的設定
     - df_path： 要預測的資料集
     - ensemble： 會將裡面的所有預測結果Ensemble
          - model_path： 模型儲存之資料夾 `ex：./save/save_kyu/kyu_time_5_input_v4_move_pred_mix_v2_augment_1_0`
          - TTA： `True/False` 是否進行TTA

# 2023AICUPGO-Identification
