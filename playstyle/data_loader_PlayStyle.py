import tensorflow as tf
import numpy as np
from go_game import capture_piece, liberty
chars = 'abcdefghijklmnopqrst'
coordinates = {k:v for v,k in enumerate(chars)}
color_map = {'B':1, 'W':-1}

BLACK = 1
WHITE = -1
EMPTY = 0
               

def inputs_liberty(board, last_move):
    """
        channel 0: board(1,0,-1)
        channel 1: empty
        channel 2: liberty of black
        channel 3: liberty of white
        channel 4: last move
    """
    inputs = np.zeros((19,19,5), dtype=np.float32)
    inputs[:,:,0] = board[:,:,0] # 
    inputs[:,:,1] = board[:,:,0] == EMPTY
    inputs[:,:,2] = board[:,:,1]
    inputs[:,:,3] = board[:,:,2]
    
    x, y, color = last_move
    inputs[y, x, 4] = color
    
    return inputs

def inputs_no_liberty(board, last_move):
    """
        channel 0: black
        channel 1: white
        channel 2: empty
        channel 3: last move
    """
    inputs = np.zeros((19,19,4), dtype=np.float32)
    inputs[:,:,0] = board[:,:,0] == BLACK
    inputs[:,:,1] = board[:,:,0] == WHITE
    inputs[:,:,2] = board[:,:,0] == EMPTY
    
    x, y, color = last_move
    inputs[y, x, 3] = color
    
    return inputs

def py_game2inputs(game, n_last_move, input_type):
    """
        input_type: 'with_liberty' or 'no_liberty'
    """
    
    if input_type == 'with_liberty':
        get_inputs = inputs_liberty
    elif input_type == 'no_liberty':
        get_inputs = inputs_no_liberty
    
    game = game.numpy().decode('ascii')
    moves_list = game.split(',')
    n_last_move = n_last_move.numpy()
 
    move_len = len(moves_list)
    
    last_n_boards = [] # 最後 n 步棋的盤面
    last_n_moves  = [] # 最後 n 步棋的位置
    
    board = np.zeros((19,19,3),dtype=np.float32)
    for move_idx in range(move_len):   
        
        move = moves_list[move_idx]# move: 'B[pq]'
        color = color_map[move[0]]
        pos_x = coordinates[move[2]]
        pos_y = coordinates[move[3]]
        
        board[pos_y, pos_x, 0] = color
        board = capture_piece(board, pos_x, pos_y, color)# 進行提子，如果有的話
        if input_type == 'with_liberty':
            board = liberty(board, pos_x, pos_y, color) # 更新氣
        # 
        if move_idx >= move_len - n_last_move:
            last_n_boards.append(board.copy())
            last_n_moves.append((pos_x, pos_y, color))
    
    # 將最後 n 步棋轉換為 input
    real_n_last_move = len(last_n_boards)
    for i in range(real_n_last_move):   
        last_n_boards[i] = get_inputs(last_n_boards[i], last_n_moves[i])
        
    # 如果實際的步數小於 n_last_move，則補上空的盤面，值為-2，為了和實際棋盤的值做區分
    if real_n_last_move < n_last_move:
        none_board = np.ones(last_n_boards[0].shape, dtype=np.float32) * -2
        last_n_boards = [none_board.copy() for i in range(n_last_move - real_n_last_move)] + last_n_boards 

    inputs = np.concatenate(last_n_boards, axis=-1)
  
    return inputs

def py_play_style_aug(games):
    games = games.numpy()
    # random rotate
    aug_type = np.random.randint(0, 4, size=1)[0]
    games = np.rot90(games, k=aug_type, axes=(-2,-3))
    # random flip
    if np.random.randint(0, 2, size=1) == 1:
        games = np.flip(games, axis=-2)
        
    return games

def py_random_pos_aug(games, n, n_last_move, n_channels):
    """
        n: 表示黑白各加0~n個棋子
        last_move: 表示是取多少步的棋盤
        目前只能在inputs為"inputs_no_liberty"使用
    """
    
    games = games.numpy()
    n_last_move = n_last_move.numpy()
    n_channels = n_channels.numpy()
    
    black_channels    = [i*n_channels + 0 for i in range(n_last_move)]
    white_channels    = [i*n_channels + 1 for i in range(n_last_move)]
    empty_channel     = [i*n_channels + 2 for i in range(n_last_move)]
    lastmove_channel  = [i*n_channels + 3 for i in range(n_last_move)]
    
    if n > 1:
        # 隨機增加0~n個棋子
        n = np.random.randint(0, n+1, size=1)[0]
    if n > 0:   
        # 紀錄原本最後n_last_move步的棋子位置
        org_last_move = games[:,:,:,lastmove_channel]
        org_last_move = np.abs(org_last_move)
        org_last_move = np.reshape(org_last_move, (-1, 19*19, n_last_move))
        org_last_move = np.transpose(org_last_move, (0, 2, 1))
        org_last_move = np.argmax(org_last_move, axis=-1)
      
        for game_ind in range(games.shape[0]):
            valid_pos = [i for i in range(361) if i not in org_last_move[game_ind]]
            new_pos = np.random.choice(valid_pos, size=2*n, replace=False)
            new_posb = new_pos[:n]
            new_posw = new_pos[n:]
            
            for i in range(n):
                new_by, new_bx = divmod(new_posb[i], 19)
                new_wy, new_wx = divmod(new_posw[i], 19)
                
                games[game_ind, new_by, new_bx, black_channels] = 1
                games[game_ind, new_wy, new_wx, white_channels] = 1
                games[game_ind, new_by, new_bx, empty_channel] = 0
                games[game_ind, new_wy, new_wx, empty_channel] = 0
                               
    return games

def game2inputs(games, n_last_move, input_type):
    """
        input_type: 'with_liberty' or 'no_liberty'
    """
    return tf.py_function(func=py_game2inputs, inp=[games, n_last_move, input_type], Tout=tf.float32)

def play_style_aug(games):
    return tf.py_function(func=py_play_style_aug, inp=[games], Tout=tf.float32)

def random_pos_aug(games, n, n_last_move, n_channels):
    return tf.py_function(func=py_random_pos_aug, inp=[games, n, n_last_move, n_channels], Tout=tf.float32)
