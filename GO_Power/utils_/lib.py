import tensorflow as tf
from tensorflow_addons.optimizers import AdamW
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import os, random, yaml, shutil
import numpy as np
from preprocess.dataloader import pre_8_augmentation

chars = 'abcdefghijklmnopqrst'
chartonumbers = {k:v for k,v in enumerate(chars)}

ROT_0 = 0
ROT_90 = 1
ROT_180 = 2
ROT_270 = 3

def save_train_yaml(config, save_path):
    # 存成.yaml檔
    with open(save_path + "/config.yaml", "w") as f:
        model_config = {
            "input_data": config["TRAINING"]["input_data"],
            "model": save_path + "/model.keras",
            "weight_path": save_path + "/best_acc.hdf5",
        }
        yaml.dump(model_config, f)

def train_model(model, train_dataset, valid_dataset, save_path="save", config=None, train_steps=None, valid_steps=None):
    # create save folder
    count = 0
    for i in range(10):
        if os.path.exists(f"{save_path}_{str(i)}"):
            # 沒有權重 就刪除整個資料夾
            if not os.path.exists(f"{save_path}_{str(i)}/best_acc.hdf5"):
                try:
                    shutil.rmtree(f"{save_path}_{str(i)}")
                except: pass
                count = i
                break
            else:
                count = i
        else:
            break

    os.makedirs(f"{save_path}_{str(count)}")
    save_path = f"{save_path}_{str(count)}"

    try:
        os.makedirs(save_path + "/tensorboard")
    except: pass

    # 儲存模型
    model.save(save_path + "/model.keras")

    # 儲存train的參數
    save_train_yaml(config, save_path)

    # callbacks
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                            patience=4, mode='auto', verbose=0, cooldown=0,
                            min_lr=5e-4)
    
    tf_callback = tf.keras.callbacks.TensorBoard(log_dir=save_path + "/tensorboard", 
                                                 histogram_freq=1)
                            
    best_loss_check = tf.keras.callbacks.ModelCheckpoint(filepath=save_path + "/best_loss.hdf5",
                            save_weights_only=True,
                            monitor='val_loss',
                            mode='min',
                            verbose=1,
                            save_best_only=True)
    
    best_acc_check = tf.keras.callbacks.ModelCheckpoint(filepath=save_path + "/best_acc.hdf5",
                            save_weights_only=True,
                            monitor='val_acc',
                            mode='max',
                            verbose=1,
                            save_best_only=True)
                            
    model.fit(
            train_dataset,
            steps_per_epoch=train_steps,
            epochs=config['TRAINING']['epochs'],
            callbacks=[
                reduce_lr, tf_callback,
                best_loss_check, best_acc_check],
            validation_data=valid_dataset,
            validation_steps=valid_steps)

    return model

def number_to_char(number):
    """
    位置轉換成字母 361 -> 'ss', 0 -> 'aa'
    """
    y, x = divmod(number, 19)

    return chartonumbers[x] + chartonumbers[y]

def top_5_preds_with_chars(predictions):
    """
    找出預測結果中前五個最大的值的index，並轉換成字母
    """
    # 找出前五個最大的值的index
    resulting_preds_numbers = [np.argpartition(prediction, -5)[-5:] for prediction in predictions]
    # 先對前五個最大的值的index做排序
    resulting_preds_numbers = [resulting_preds_number[np.argsort(predictions[i][resulting_preds_number])[::-1]] for i, resulting_preds_number in enumerate(resulting_preds_numbers)]

    resulting_preds_chars = np.vectorize(number_to_char)(resulting_preds_numbers)
    return resulting_preds_chars


def pre_to_board(pre):
    """
    Label轉換成board, (,361) -> (19, 19)
    """
    board = np.zeros((19, 19), dtype=np.float32)
    for i in range(361):
        y, x = divmod(i, 19)
        board[y][x] = pre[i]
    
    return board

def turn_label_to_original(pres, aug_pipe):
    """
    將TTA過的label轉回原本的樣子
    """
    # 轉成棋盤
    pres = np.reshape(pres, (-1, 19, 19))
    # print(board.shape)
    # 依照aug_pipe轉回原本的樣子
    pres = pre_8_augmentation(pres, aug_pipe, label_mode=True)
    # 轉回一維
    pres = np.reshape(pres, (-1, 19*19))

    return pres