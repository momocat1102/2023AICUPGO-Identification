import numpy as np
import tensorflow as tf
from utils_.go_game import capture_piece, liberty

chars = 'abcdefghijklmnopqrst'
coordinates = {k:v for v,k in enumerate(chars)}

BLACK = 1
WHITE = -1

def input_board(moves_list, target_player, org_board, move_data=None, input_data=False, input_type=None):
    """
    將位置轉換成棋盤
    """
    board = org_board.copy()

    if input_type == 'baseline':
        """
        channel 0: 黑子 (1: black, 0: other)
        channel 1: 白子 (1: white, 0: other)
        channel 2: 沒落子的位置 (1: empty, 0: have stone)
        channel 3: 最後一手的位置
        """
        if not input_data:
            if move_data[0] == BLACK:
                board[move_data[2], move_data[1], 0] = 1
            if move_data[0] == WHITE:
                board[move_data[2], move_data[1], 1] = 1
            board[move_data[2], move_data[1], 2] = 1
        else:
            board[:, :, 2] = np.where(board[:, :, 2] == 0, 1, 0)
            if len(moves_list) > 0:
                last_move_x = coordinates[moves_list[-1][2]]
                last_move_y = coordinates[moves_list[-1][3]]
                board[last_move_y, last_move_x, 3] = 1

    elif input_type == 'inputs1':
        """
        channel 0: color (-1: black, 1: white 0: empty)
        channel 1: color 下棋順序
        channel 2: 要下的顏色, 用可行步表示(-1: black, 1: white)
        channel 3: 最後一手的位置
        """
        if not input_data:
            color = move_data[0]
            x = move_data[1]
            y = move_data[2]
            board[y, x, 0] = color
            board[y, x, 1] = len(moves_list) / 361
            board[y, x, 2] = 1
        else:
            board[:, :, 2] = np.where(board[:, :, 2] == 0, 1, 0)*target_player
            if len(moves_list) > 0:
                last_move_x = coordinates[moves_list[-1][2]]
                last_move_y = coordinates[moves_list[-1][3]]
                board[last_move_y, last_move_x, 3] = 1

    elif input_type == 'input_v2':
        """
        channel 0: 白子 (1: white, 0: other)
        channel 1: 黑子 (1: black, 0: other)
        channel 2: 有下棋的地方 (1: 有下棋, 0: 沒下棋)
        channel 3: 下棋順序        
        """
        if not input_data:
            color = move_data[0]
            x = move_data[1]
            y = move_data[2]
            if color == WHITE:
                board[y, x, 0] = 1
            else:
                board[y, x, 1] = 1
            board[y, x, 2] = 1
            board[y, x, 3] = len(moves_list) / 361
        else:
            board[:, :, 0:3] = board[:, :, 0:3] + 1

    elif input_type == 'time_input_v2':
        """
        shape = (19, 19, 4*3)
        channel 0: 白子 (1: white, 0: other)
        channel 1: 黑子 (1: black, 0: other)
        channel 2: 有下棋的地方 (1: 有下棋, 0: 沒下棋)
        channel 3: 下棋順序        
        """
        if not input_data:
            for i in range(2, 0, -1):
                for j in range(4):
                    # 後移到下一個時間點
                    # print(i*4 - j + 5, i*4 - j - 1)
                    board[:, :, i*4 - j + 3] = board[:, :, i*4 - j - 1]
            color = move_data[0]
            x = move_data[1]
            y = move_data[2]
            if color == WHITE:
                board[y, x, 0] = 1
            else:
                board[y, x, 1] = 1
            board[y, x, 2] = 1
            board[y, x, 3] = len(moves_list) / 361
        else:
            pass

    elif input_type == 'input_v3':
        """
        shape = (19, 19, 6)
        channel 0: 棋盤 (1: black, -1: white, 0: empty)
        channel 1: 空白 (1: empty, 0: other)
        channel 2: 下棋順序
        channel 3: 黑子的氣 (1: 有氣, 0: 沒氣)
        channel 4: 白子的氣 (1: 有氣, 0: 沒氣)
        channel 5: 最後一手的位置
        """
        if not input_data:
            color = move_data[0]
            x = move_data[1]
            y = move_data[2]
            board[y, x, 0] = color
            board = capture_piece(board, x, y, color)
            board = liberty(board, x, y, color)
            board[y, x, 2] = len(moves_list) / 361
        else:
            # 找出空白的位置
            board[:, :, 1] = np.where(board[:, :, 0] == 0, 1, 0)
            if len(moves_list) > 0:
                last_move_x = coordinates[moves_list[-1][2]]
                last_move_y = coordinates[moves_list[-1][3]]
                board[last_move_y, last_move_x, 5] = 1

    elif input_type == 'time_input_v3':
        """
        shape = (19, 19, 6*3)
        channel 0*i: 棋盤 (1: black, -1: white, 0: empty)
        channel 1*i: 空白 (1: empty, 0: other)
        channel 2*i: 下棋順序
        channel 3*i: 黑子的氣 (1: 有氣, 0: 沒氣)
        channel 4*i: 白子的氣 (1: 有氣, 0: 沒氣)
        channel 5*i: 最後一手的位置
        """
        if not input_data:
            # [0, 1, 2, 3, 4, 5,| 6, 7, 8, 9, 10, 11,| 12, 13, 14, 15, 16, 17]
            for i in range(2, 0, -1):
                for j in range(6):
                    # 後移到下一個時間點
                    # print(i*6 - j - 1, i*6 - j + 5)
                    board[:, :, i*6 - j + 5] = board[:, :, i*6 - j - 1]
            color = move_data[0]
            x = move_data[1]
            y = move_data[2]
            board[y, x, 0] = color
            board = capture_piece(board, x, y, color)
            board = liberty(board, x, y, color)
            board[y, x, 2] = len(moves_list) / 361
            if color == BLACK:
                board[:, :, 5] = 1
            else:
                board[:, :, 5] = -1
            board[:, :, :6] = board[:, :, :6]

    elif input_type == 'time_5_input_v4':
        """
        shape = (19, 19, 5*5)
        channel 0*i: 棋盤 (1: black, -1: white, 0: empty)
        channel 1*i: 有下棋的地方 (1: 有下棋, 0: 沒下棋)
        channel 2*i: 最後一手的位置
        channel 3*i: 黑子的氣 (1: 有氣, 0: 沒氣)
        channel 4*i: 白子的氣 (1: 有氣, 0: 沒氣)
        """
        n = 5 # channel數量
        if not input_data:
            # [0, 1, 2, 3, 4, 5,| 6, 7, 8, 9, 10, 11,| 12, 13, 14, 15, 16, 17]
            for i in range(4, 0, -1):
                for j in range(n):
                    # 後移到下一個時間點
                    # print(i*6 - j - 1, i*6 - j + 5)
                    board[:, :, i*n - j + n - 1] = board[:, :, i*n - j - 1]
            color = move_data[0]
            x = move_data[1]
            y = move_data[2]
            board[y, x, 0] = color
            board = capture_piece(board, x, y, color)
            board = liberty(board, x, y, color)
            board[y, x, 1] = 1
            if color == BLACK:
                board[:, :, 2] = 1
            else:
                board[:, :, 2] = -1
            # board[:, :, :6] = board[:, :, :6]

    elif input_type == 'time_7_input_v4':
        """
        shape = (19, 19, 5*7)
        channel 0*i: 棋盤 (1: black, -1: white, 0: empty)
        channel 1*i: 有下棋的地方 (1: 有下棋, 0: 沒下棋)
        channel 2*i: 最後一手的位置
        channel 3*i: 黑子的氣 (1: 有氣, 0: 沒氣)
        channel 4*i: 白子的氣 (1: 有氣, 0: 沒氣)
        """
        n = 5 # channel數量
        if not input_data:
            # [0, 1, 2, 3, 4, 5,| 6, 7, 8, 9, 10, 11,| 12, 13, 14, 15, 16, 17]
            for i in range(6, 0, -1):
                for j in range(n):
                    # 後移到下一個時間點
                    # print(i*6 - j - 1, i*6 - j + 5)
                    board[:, :, i*n - j + n - 1] = board[:, :, i*n - j - 1]
            color = move_data[0]
            x = move_data[1]
            y = move_data[2]
            board[y, x, 0] = color
            board = capture_piece(board, x, y, color)
            board = liberty(board, x, y, color)
            board[y, x, 1] = 1
            if color == BLACK:
                board[:, :, 2] = 1
            else:
                board[:, :, 2] = -1
            # board[:, :, :6] = board[:, :, :6]

    elif input_type == 'time_3_input_v4':
        """
        shape = (19, 19, 5*3)
        channel 0*i: 棋盤 (1: black, -1: white, 0: empty)
        channel 1*i: 有下棋的地方 (1: 有下棋, 0: 沒下棋)
        channel 2*i: 最後一手的位置
        channel 3*i: 黑子的氣 (1: 有氣, 0: 沒氣)
        channel 4*i: 白子的氣 (1: 有氣, 0: 沒氣)
        """
        n = 5 # channel數量
        if not input_data:
            # [0, 1, 2, 3, 4, 5,| 6, 7, 8, 9, 10, 11,| 12, 13, 14, 15, 16, 17]
            for i in range(2, 0, -1):
                for j in range(n):
                    # 後移到下一個時間點
                    # print(i*6 - j - 1, i*6 - j + 5)
                    board[:, :, i*n - j + n - 1] = board[:, :, i*n - j - 1]
            color = move_data[0]
            x = move_data[1]
            y = move_data[2]
            board[y, x, 0] = color
            board = capture_piece(board, x, y, color)
            board = liberty(board, x, y, color)
            board[y, x, 1] = 1
            if color == BLACK:
                board[:, :, 2] = 1
            else:
                board[:, :, 2] = -1

    return board



