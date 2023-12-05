import numpy as np
from preprocess.dataloader import decode_game2board, pre_aug_board
import tensorflow as tf
from utils_.lib import top_5_preds_with_chars, turn_label_to_original
import yaml, os

with open('predict.yaml', 'r') as f:
    config = yaml.safe_load(f)

def predict_data(config_type):
    """
    用於預測的函式
    config_type: dict, 用於指定要預測的資料
    """
    # load csv
    print(config_type)
    df = open(config_type["df_path"]).read().splitlines()
    players = [i.split(',',2)[1] for i in df]
    games = [i.split(',',2)[2] for i in df]
    pre = np.zeros((len(df), 361), dtype=np.float32)

    # Ensemble
    for model_info in config_type["ensemble"]:
        # load model
        with open(f'{model_info["model_path"]}/config.yaml', 'r') as f:
            info = yaml.safe_load(f)
        model = tf.keras.models.load_model(info["model"])
        model.load_weights(info["weight_path"])
        print(len(df))
        
        # TTA
        if model_info["TTA"] != False:            
            test_dataset = tf.data.Dataset.from_tensor_slices((games, players))
            test_dataset = test_dataset.map(lambda x,y: decode_game2board(x, y, input_data=info["input_data"], to_fit=False)).unbatch().batch(1024)
            test_dataset = test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

            for i, aug_pipe in enumerate([[0, 0], [1, 0], [2, 0], [3, 0], [0, 1], [1, 1], [2, 1], [3, 1]]):
                print("TTA steps:" + str(i + 1))
                # aug_pipe = [np.random.randint(0, 4), np.random.randint(0, 2)]
                test_dataset_tta = test_dataset.map(lambda x: pre_aug_board(x, aug_pipe)) 
                pres = model.predict(test_dataset_tta)
                pres = turn_label_to_original(pres, aug_pipe)
                pre += pres
        else:
            # load dataset
            test_dataset = tf.data.Dataset.from_tensor_slices((games, players))
            test_dataset = test_dataset.map(lambda x,y: decode_game2board(x, y, input_data=info["input_data"], to_fit=False))
            test_dataset = test_dataset.unbatch().batch(1024)
            test_dataset = test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            pre += model.predict(test_dataset)

    return df, pre

def main():

    if config["PREDICT"]["kyu_predict"]:
        # 級的訓練
        df_kyu, kyu_datas = predict_data(config["PREDICT"]["kyu"])
        pre_chars_kyu = top_5_preds_with_chars(kyu_datas)

        with open('./result/result_kyu.csv', 'w') as f:
            for i in range(len(pre_chars_kyu)):
                f.write(df_kyu[i].split(',')[0] + ',' + ','.join(pre_chars_kyu[i]) + '\n')

    if config["PREDICT"]["dan_predict"]:
        # 段的訓練
        df_dan, dan_datas = predict_data(config["PREDICT"]["dan"])
        pre_chars_dan = top_5_preds_with_chars(dan_datas)

        with open('./result/result_dan.csv', 'w') as f:
            for i in range(len(pre_chars_dan)):
                f.write(df_dan[i].split(',')[0] + ',' + ','.join(pre_chars_dan[i]) + '\n')

def add_csv(csvs):
    with open('./result/result.csv', 'w') as f:
        for csv in csvs:
            for line in open(csv).read().splitlines():
                f.write(line + '\n')

if __name__ == '__main__':
    main()
    try:
        add_csv(config["PREDICT"]["csvs"])
    except:
        print('add_csv error')