from preprocess.dataloader import decode_game2board, augment_board, balance_board
from utils_.lib import train_model
import tensorflow as tf
from tensorflow_addons.optimizers import AdamW
from model.create_model import models
import yaml, os, numpy as np

with open('train.yaml', 'r') as f:
    config = yaml.safe_load(f)

to_fit = True

# 由data_analyse.py算出，平衡過的大約的資料數量
DAN_DATA_NUM = 9348765
KYU_DATA_NUM = 11369833

# 生成資料夾
try:
    os.makedirs(config["TRAINING"]["cache_path"])
except: pass

try:
    os.mkdir("./save")
except: pass
try:
    os.mkdir("./save/save_dan")
except: pass
try:
    os.mkdir("./save/save_kyu")
except: pass

def train():
    # load data
    df = open(f'./CSVs/{config["TRAINING"]["power"]}_train.csv').read().splitlines()
    rank = 0
    data_num = DAN_DATA_NUM
    if config["TRAINING"]["power"] == "kyu":
        rank = 1
        data_num = KYU_DATA_NUM
    np.random.seed(1634)
    np.random.shuffle(df)
    np.random.seed(None)
    players = [i.split(',',2)[1] for i in df]
    games = [i.split(',',2)[2] for i in df]

    dataset = tf.data.Dataset.from_tensor_slices((games, players))

    # split data into train and validation sets
    total_size = dataset.cardinality().numpy()
    if config["TRAINING"]["train_status"] == '1': # 拿最後的資料當作validation
        train_size = int(total_size * (1 - config["TRAINING"]["val_ratio"]))
        train_dataset = dataset.take(train_size)
        val_dataset = dataset.skip(train_size)
    else: # 拿最前面的資料當作validation
        val_size = int(total_size * config["TRAINING"]["val_ratio"])
        train_dataset = dataset.skip(val_size)
        val_dataset = dataset.take(val_size)

    # complite dataset
    train_dataset = train_dataset.map(lambda x,y: decode_game2board(x, y, input_data=config["TRAINING"]["input_data"]), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    train_dataset = train_dataset.cache(config["TRAINING"]["cache_path"] + "/train")
    train_dataset = train_dataset.map(lambda x,y,z: balance_board(x, y, z, rank=rank), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # augmentation
    if config["TRAINING"]["augment"] != "org":
        if config["TRAINING"]["augment"] == "1":
            train_dataset = train_dataset.map(lambda x,y: augment_board(x, y, aug_pipe=1), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        elif config["TRAINING"]["augment"] == "8":
            # 八個相位的augmentation
            for aug_pipe in [[0, 0], [1, 0], [2, 0], [3, 0], [0, 1], [1, 1], [2, 1], [3, 1]]:
                train_dataset_aug = train_dataset.map(lambda x,y: augment_board(x, y, aug_pipe), num_parallel_calls=tf.data.experimental.AUTOTUNE)
                train_dataset = train_dataset.concatenate(train_dataset_aug)
    
    train_dataset = train_dataset.unbatch().shuffle(buffer_size=1000, seed=config["TRAINING"]["shuffle_seed"]).batch(config["TRAINING"]["batch_size"])
    train_dataset = train_dataset.map(lambda x,y: (x, tf.reshape(y, (-1, 19*19))))
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE).repeat()
    
    # <--------------------------------------------------validation---------------------------------------------------------------------->

    # complite dataset
    val_dataset = val_dataset.map(lambda x,y: decode_game2board(x, y, input_data=config["TRAINING"]["input_data"]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.cache(config["TRAINING"]["cache_path"] + "/val")
    val_dataset = val_dataset.map(lambda x,y,z: balance_board(x, y, z, rank=rank), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # augmentation
    if config["TRAINING"]["augment"] != "org":
        if config["TRAINING"]["augment"] == "1":
            val_dataset = val_dataset.map(lambda x,y: augment_board(x, y, aug_pipe=1), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            # 八個相位的augmentation
            for aug_pipe in [[0, 0], [1, 0], [2, 0], [3, 0], [0, 1], [1, 1], [2, 1], [3, 1]]:
                val_dataset_aug = val_dataset.map(lambda x,y: augment_board(x, y, aug_pipe), num_parallel_calls=tf.data.experimental.AUTOTUNE)
                val_dataset = val_dataset.concatenate(val_dataset_aug)

    val_dataset = val_dataset.unbatch().batch(config["TRAINING"]["batch_size"])
    val_dataset = val_dataset.map(lambda x,y: (x, tf.reshape(y, (-1, 19*19))))
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE).repeat()

    # # choose model
    input_shape = tuple(map(int, config['TRAINING']['input_data'].split(':')[1].split(',')))
    model = models[config['TRAINING']['model']](input_shape)
    model.summary()

    try:
        model.load_weights(config['TRAINING']['weight_path'], by_name=True, skip_mismatch=True)
    except:
        print("No weight file")

    # tf.keras.utils.plot_model(model, "transformer_model.png", show_shapes=True)

    # training
    model.compile(
        optimizer=AdamW(learning_rate=0.001, weight_decay=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="acc"),
            tf.keras.metrics.TopKCategoricalAccuracy(name="top_5_acc", k=5)]
    )
    if not config["TRAINING"]["evaluate"]:
        input_type = config["TRAINING"]["input_data"].split(':')[0]
        if config["TRAINING"]["augment"]:
            save_path = f'./save/save_{config["TRAINING"]["power"]}/{config["TRAINING"]["power"]}_\
{input_type}_{config["TRAINING"]["model"]}_augment_{config["TRAINING"]["train_status"]}'
        else:
            save_path = f'./save/save_{config["TRAINING"]["power"]}/{config["TRAINING"]["power"]}_\
{input_type}_{config["TRAINING"]["model"]}_{config["TRAINING"]["train_status"]}'
        print(save_path)
        model = train_model(model, train_dataset, 
                            val_dataset,
                            save_path=save_path,
                            config=config, 
                            train_steps=int((data_num * (1 - config["TRAINING"]["val_ratio"])) // config["TRAINING"]["batch_size"]),
                            valid_steps=int((data_num * config["TRAINING"]["val_ratio"]) // config["TRAINING"]["batch_size"])
                            )
        # predict
        # model.load_weights(f'{save_path}/best_acc.hdf5')
        model.evaluate(val_dataset)
    else:
        model.evaluate(val_dataset)
    
if __name__ == '__main__':
    train()
