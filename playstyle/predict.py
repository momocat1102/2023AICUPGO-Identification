import tensorflow as tf
import numpy as np
from data_loader_PlayStyle import play_style_aug, random_pos_aug, game2inputs
from models import  ps_cnn_atte

import os, shutil, glob

np.set_printoptions(threshold=np.inf, linewidth=150)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

config = {'save_dir':'./saves/play_style', # 權重的資料夾
          'weight_name':'ps_cnn_atte',     # 儲存權重的名稱
          'result_dir':'./results',        # 儲存預測結果的資料夾
          'n_models':1, 
          'epochs':20,
          'batch_size':512,
          'n_last_move':3, # 設定input要取到前幾手
          'input_type':'with_liberty', #'with_liberty' or 'no_liberty'
          'n_tta':16, # 設定要TTA次數
          }

if config['input_type'] == 'with_liberty':
    config['n_channels'] = 5
elif config['input_type'] == 'no_liberty':  
    config['n_channels'] = 4
    
config['weight_name'] += f"L{config['n_last_move']}_{config['input_type']}"

def pred_play_style():
    shutil.rmtree('./caches/test_cache/', ignore_errors=True)
    os.makedirs('./caches/test_cache/')
    
    # 讀取要預測的資料
    df = open('./CSVs/play_style_test_pr+pu.csv').read().splitlines()
    id = [i.split(',',1)[0] for i in df]
    games = [i.split(',',1)[1] for i in df]
    # 建立dataset
    dataset = tf.data.Dataset.from_tensor_slices((games))
    dataset = dataset.map(lambda x: game2inputs(x, n_last_move=config['n_last_move'], input_type=config['input_type']), -1)
    dataset = dataset.cache('caches/test_cache/play_style')
    # 要使用TTA，所以還是要做augmentation
    dataset = dataset.map(play_style_aug, -1)
    dataset = dataset.batch(1024)
    #dataset = dataset.map(lambda x: random_pos_aug(x, n=3, n_last_move=n_last_move,n_channels=n_channels), -1).prefetch(-1)

    # 載入模型和權重
    models = [ps_cnn_atte(n_last_move=config['n_last_move'], n_channels=config['n_channels']) for i in range(config['n_models'])]
    for i in range(config['n_models']):
        models[i].load_weights(f"{config['save_dir']}/best_{config['weight_name']}_fold{i}_acc.h5")
    
    total = [np.zeros((len(id),3)) for i in range(config['n_models'])]
    # 開始預測
    for i in range(config['n_tta']):
        print(i)
        for j in range(config['n_models']):
            pred = models[j].predict(dataset,verbose=1)
            total[j] += pred
            
    # 個別權重預測結果
    preds = [np.argmax(total[i], axis=-1) + 1 for i in range(config['n_models'])]
    
    # Ensemble
    pred_total = np.sum(total, axis=0)
    pred_total = np.argmax(pred_total, axis=-1) + 1
    
    # 檢查是否有相同名稱的資料夾，若有則在名稱後面加上數字
    same_path = glob.glob(f'{config["result_dir"]}/{config["weight_name"]}_*')
    length = len(same_path)
        
    os.makedirs(f'{config["result_dir"]}/{config["weight_name"]}_{length}/', exist_ok=False)
    
    # 儲存Ensemble的預測結果
    with open(f'{config["result_dir"]}/{config["weight_name"]}_{length}/pred_ensemble.csv', 'w') as f:
        for i in range(len(id)):
            f.write(id[i] + ',' + str(pred_total[i]) + '\n')   
    # 儲存個別權重的預測結果   
    for i in range(config['n_models']):
        with open(f'{config["result_dir"]}/{config["weight_name"]}_{length}/pred_fold{i}.csv', 'w') as f:
            for j in range(len(id)):
                f.write(id[j] + ',' + str(preds[i][j]) + '\n')
    
    
if __name__ == '__main__':
    pred_play_style()
