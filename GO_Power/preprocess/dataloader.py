from preprocess.input_type import input_board
import numpy as np
import tensorflow as tf

color_map = {'B':1, 'W':-1}
chars = 'abcdefghijklmnopqrst'
coordinates = {k:v for v,k in enumerate(chars)}

ROT_0 = 0
ROT_90 = 1
ROT_180 = 2
ROT_270 = 3

dan_balance = np.load('./balance_dan.npy')
kyu_balance = np.load('./balance_kyu.npy')


def data_augmentation(boards, labels, aug_pipe):
    """
    訓練時的擴增，作盤面的旋轉與翻轉
    aug_pipe: list, 用於指定要做哪些augmentation [rotate_idx, flip_lr_idx]
    """
    boards = boards.numpy()
    labels = labels.numpy()
    aug_pipe = aug_pipe.numpy()
    if aug_pipe == 1:
        aug_pipe = [np.random.randint(0, 4), np.random.randint(0, 2)]
    
    boards = np.rot90(boards, k=aug_pipe[0], axes=(1,2))
    labels = np.rot90(labels, k=aug_pipe[0], axes=(1,2))
    if aug_pipe[1] == 1:
        boards = np.flip(boards, axis=1)
        labels = np.flip(labels, axis=1)
        
    return boards, labels

def pre_8_augmentation(boards, aug_pipe, label_mode=False):
    """
    預測時的擴增，作盤面的旋轉與翻轉
    aug_pipe: list, 用於指定要做哪些augmentation [rotate_idx, flip_lr_idx]
    label_mode: bool, 將Label轉換回原本的樣子
    """
    
    if not label_mode:
        # print(total_tta_list)
        boards = boards.numpy()
        aug_pipe = aug_pipe.numpy()
        boards = np.rot90(boards, k=aug_pipe[0], axes=(1,2))
        if aug_pipe[1] == 1:
            boards = np.flip(boards, axis=1)
    else:
        if aug_pipe[1] == 1:
            boards = np.flip(boards, axis=1)
        boards = np.rot90(boards, k=-aug_pipe[0], axes=(1,2))
        
    return boards

def to_move_pred_board(game, target_player, input_data, to_fit=True):
    """
    將棋譜轉換為盤面
    game: str, 棋譜
    target_player: str, 目標玩家
    input_data: str, 選擇的input格式，包含input_type與input_shape
    to_fit: bool, 是否要輸出label
    """

    labels = []
    boards = []
    move_idxs = [] # 紀錄該殘局的長度，用於balance

    target_player = target_player.numpy().decode('ascii') # 將 tensorflow 張量轉換為字串
    target_player = color_map[target_player] # 將字串轉換為數字，B -> 1, W -> -1
    game = game.numpy().decode('ascii')

    moves_list = game.split(',')
    move_len = len(moves_list)
    # print(type(input_data), input_data)
    input_data = input_data.numpy().decode('ascii')
    input_shape = input_data.split(':')[1]
    input_type = input_data.split(':')[0]
    shape = tuple(map(int, input_shape.split(',')))
    board = np.zeros(shape, dtype=np.float32)

    for move_idx in range(move_len):
        move = moves_list[move_idx] # move: 'B[pq]'
        color = color_map[move[0]]
        pos_x = coordinates[move[2]]
        pos_y = coordinates[move[3]]
        move_data = [color, pos_x, pos_y]

        if color == target_player and to_fit and move_idx != 0: # 如果該步棋是目標玩家下的
            label = np.zeros((19,19), dtype=np.float32)
            label[pos_y, pos_x] = 1
            boards.append(input_board(moves_list[:move_idx], target_player, board, None, input_data=True, input_type=input_type))
            labels.append(label) # 將目標玩家下的棋子在棋盤上的位置加入 labels
            move_idxs.append(move_idx)

        board = input_board(moves_list[:move_idx + 1], target_player, board, move_data, input_type=input_type)     
    if not to_fit:
        boards.append(input_board(moves_list, target_player, board, None, input_data=True, input_type=input_type))
        
    boards = np.array(boards, dtype=np.float32)
    
    if to_fit:
        labels = np.array(labels, dtype=np.int32) 
        return boards, labels, move_idxs
    
    return boards, 0, 0

def to_balance_board(boards, labels, move_idxs, rank):
    """
    平衡資料，將少的資料多複製，多的資料隨機選擇是否丟棄
    boards: np.array, 盤面
    labels: np.array, 標籤
    move_idxs: np.array, 殘局長度
    rank: int, 0為dan, 1為kyu
    """
    select_boards = []
    select_labels = []
    boards = boards.numpy()
    labels = labels.numpy()
    move_idxs = move_idxs.numpy()
    balance_weight = dan_balance if rank == 0 else kyu_balance
    for i, move_idx in enumerate(move_idxs):
        # 小於1的話，就以balance_weight[move_idx]的機率丟棄
        if balance_weight[move_idx] < 1 and np.random.rand() < balance_weight[move_idx]:
            select_boards.append(boards[i])
            select_labels.append(labels[i])
        elif balance_weight[move_idx] >= 1:
            integer = balance_weight[move_idx] // 1
            for _ in range(int(integer)):
                select_boards.append(boards[i])
                select_labels.append(labels[i])
            if np.random.rand() <= balance_weight[move_idx] - integer:
                select_boards.append(boards[i])
                select_labels.append(labels[i])

    select_boards = np.array(select_boards, dtype=np.float32)
    select_labels = np.array(select_labels, dtype=np.int32)

    return select_boards, select_labels


#@tf.function
def decode_game2board(games, players, input_data, to_fit=True):
    
    boards, labels, move_idxs = tf.py_function(func=to_move_pred_board, inp=[games, players, input_data, to_fit], Tout=[tf.float32, tf.float32, tf.int32])
    
    if to_fit:
        return boards, labels, move_idxs
    else:
        return boards
    
def balance_board(boards, labels, move_idxs, rank):
    boards, labels = tf.py_function(func=to_balance_board, inp=[boards, labels, move_idxs, rank], Tout=[tf.float32, tf.float32])

    return boards, labels
    
def augment_board(boards, labels, aug_pipe):
    
    boards, labels = tf.py_function(func=data_augmentation, inp=[boards, labels, aug_pipe], Tout=[tf.float32, tf.float32])     
    
    return boards, labels
    
def pre_aug_board(boards, aug_pipe):

    boards = tf.py_function(func=pre_8_augmentation, inp=[boards, aug_pipe], Tout=tf.float32)
    
    return boards