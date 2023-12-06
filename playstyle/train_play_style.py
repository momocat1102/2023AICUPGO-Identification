import tensorflow as tf
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

import numpy as np
import os, shutil

from data_loader_PlayStyle import play_style_aug, game2inputs, random_pos_aug
from models import ps_cnn_atte


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.set_printoptions(threshold=np.inf, linewidth=150)

config = {'save_dir':'./saves/play_style',   # 設定儲存權重的資料夾
          'weight_name':'ps_cnn_atte',     # 設定儲存權重的名稱
          "train_val_split":0.1,             # 設定驗證集的比例
          'n_models':3, # 設定要訓練幾個模型，每個模型的差異在於其train_dataset會有所不同
          'epochs':20,
          'batch_size':256,
          'lr':0.0012,
          'n_last_move':3, # 設定input要取到前幾手
          'input_type':'with_liberty' #'with_liberty' or 'no_liberty'
          }


if config['input_type'] == 'with_liberty':
    config['n_channels'] = 5
elif config['input_type'] == 'no_liberty':
    config['n_channels'] = 4
    
config['weight_name'] += f"L{config['n_last_move']}_{config['input_type']}"

def evaluate(valid_dataset, fold_id):
    print(f"fold {fold_id}")
    
    model = ps_cnn_atte(n_last_move=config['n_last_move'], n_channels=config['n_channels'])
    
    save_path = f"{config['save_dir']}/best_{config['weight_name']}_fold{fold_id}"
    model.load_weights(save_path + '_acc.h5') 
    
    #if fold_id == 0: model.summary()
    
    model.evaluate(valid_dataset, verbose=1)
    
def train(train_dataset, valid_dataset, fold_id):
    # 建立模型
    model = ps_cnn_atte(n_last_move=config['n_last_move'], n_channels=config['n_channels'], lr=config['lr'])
    
    if fold_id == 0: 
        model.summary()
        model.save(f"{config['save_dir']}/{config['weight_name']}_model.h5")
        
    save_path = f"{config['save_dir']}/best_{config['weight_name']}_fold{fold_id}"
    # Checkpoint 會儲存最高Accuracy和最小Loss的權重
    ckpt_loss = ModelCheckpoint(save_path + '_loss.h5', monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min', verbose=1)
    ckpt_acc = ModelCheckpoint(save_path + '_acc.h5', monitor='val_acc', save_best_only=True, save_weights_only=True, mode='max', verbose=1)
    # 當Loss長時間不下降時，Learning rate會以factor的比例減少
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=2, min_lr=0.0001)
    # 開始訓練
    model.fit(train_dataset,
              epochs=config['epochs'],
              validation_data=valid_dataset,
              callbacks=[ckpt_loss, ckpt_acc, reduce_lr],
              workers=5,
              verbose=1
             )
    
def main():

    os.makedirs(config['save_dir'], exist_ok=True)
    
    np.random.seed(1545) # 保證每次訓練的資料都一樣
    df = open('./CSVs/play_style_train.csv').read().splitlines()
    data_len = len(df)
    val_size = int(data_len * config['train_val_split'])
    print(f"val_size: {val_size}")
    
    # 清空cache資料夾
    shutil.rmtree('./caches/train_cache', ignore_errors=True)
    os.makedirs('./caches/train_cache', exist_ok=True)
    
    for i in range(config['n_models']):
        print(f"fold {i}")
        
        np.random.shuffle(df)
        games = [i.split(',',2)[2] for i in df]
        game_styles = [int(i.split(',',2)[1])-1 for i in df]
        
        dataset = tf.data.Dataset.from_tensor_slices((games, game_styles))
        
        # 處理input 和 label
        dataset = dataset.map(lambda x, y: (game2inputs(x,
                                                        n_last_move = config['n_last_move'],
                                                        input_type  = config['input_type']
                                                        ), y), num_parallel_calls=-1)
        dataset = dataset.map(lambda x, y: (x, tf.one_hot(y, 3, dtype=tf.float32)), num_parallel_calls=-1)
        # cache，加快讀取速度，但會佔用較多記憶體和硬碟空間
        dataset = dataset.cache(f'./caches/train_cache/train_f{i}')
        
        # 切分訓練集和驗證集
        train_dataset = dataset.skip(val_size)
        val_dataset = dataset.take(val_size)
      
        train_dataset = train_dataset.repeat(10).shuffle(1000)
        train_dataset = train_dataset.batch(config['batch_size'])
        # 資料增強(隨機旋轉+隨機翻轉)
        train_dataset = train_dataset.map(lambda x, y: (play_style_aug(x), y), num_parallel_calls=-1)
        if config['input_type'] == 'no_liberty':
            train_dataset = train_dataset.map(lambda x, y: (random_pos_aug(x,
                                                                        n=3,
                                                                        n_last_move=config['n_last_move'],
                                                                        n_channels=config['n_channels']
                                                                        ),
                                                            y), num_parallel_calls=-1)
        train_dataset = train_dataset.prefetch(-1)
        
        val_dataset = val_dataset.batch(config['batch_size']).prefetch(-1)
        
        train(train_dataset, val_dataset, i)
        #evaluate(val_dataset, i)
    
if __name__ == '__main__':
    main()
    