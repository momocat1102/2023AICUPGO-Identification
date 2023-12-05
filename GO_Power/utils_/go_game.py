import numpy as np
import tkinter as tk
EMPTY = 0
BLACK = 1
WHITE = -1
LEFT_EDGE = 0
RIGHT_EDGE = 18
TOP_EDGE = 0
BOTTOM_EDGE = 18
CAPTURE = 1
LIBERTY = 0
 
def find_connected(board, pos_x, pos_y, color, mode=CAPTURE):
    '''
    找出指定顏色所連接的區域，用於提子
    '''
    connected_pieces = []
    liberties = []
    visited = np.zeros((19, 19),dtype=np.bool_)
    stack = []
    stack.append((pos_x, pos_y))
    visited[pos_y][pos_x] = True
    dx = [0, 0, 1, -1]
    dy = [1, -1, 0, 0]
    while len(stack) > 0:
        x, y = stack.pop()
        connected_pieces.append((x, y))
        for i in range(4):
            new_x = x + dx[i]
            new_y = y + dy[i]
            if new_x < LEFT_EDGE or new_x > RIGHT_EDGE or new_y < TOP_EDGE or new_y > BOTTOM_EDGE:
                continue
            if visited[new_y][new_x]:
                continue
            if board[new_y][new_x][0] == EMPTY:
                liberties.append((new_x, new_y))
                if mode == CAPTURE:
                    break
                visited[new_y][new_x] = True
            if board[new_y][new_x][0] == color:
                stack.append((new_x, new_y))
                visited[new_y][new_x] = True
    return liberties, connected_pieces

def capture_piece(board, pos_x, pos_y, move_color):
    '''
    提子演算法，將被包圍的棋子提掉，且會重新計算周圍的氣
    channel 0: 棋盤 channel 3: 黑子的氣, channel 4: 白子的氣
    '''
    dx = [0, 0, 1, -1]
    dy = [1, -1, 0, 0]
    for i in range(4):
        x = pos_x + dx[i]
        y = pos_y + dy[i]
        if x < LEFT_EDGE or x > RIGHT_EDGE or y < TOP_EDGE or y > BOTTOM_EDGE:
            continue
        
        if board[y][x][0] == -move_color:
            liberties, connected_pieces = find_connected(board, x, y, move_color*-1)

            if len(liberties) == 0: # 沒氣
                for cx, cy in connected_pieces:
                    for i in range(4):
                        new_x = cx + dx[i]
                        new_y = cy + dy[i]
                        if new_x < LEFT_EDGE or new_x > RIGHT_EDGE or new_y < TOP_EDGE or new_y > BOTTOM_EDGE:
                            continue
                        if board[new_y][new_x][0] == move_color:
                            if move_color == BLACK:
                                board[cy][cx][3] = 1
                                break
                            else:
                                board[cy][cx][4] = 1
                                break
                    board[cy][cx][0] = EMPTY
    return board

# def liberty(board, move_color):
#     liberty_board = np.zeros((19, 19, 2))
#     visited = np.zeros((19, 19, 2),dtype=np.bool_)
#     # 找出所有黑子
#     if move_color == BLACK:
#         my_index = np.where(board == BLACK)
#         opponent_index = np.where(board == WHITE)
#     else:
#         my_index = np.where(board == WHITE)
#         opponent_index = np.where(board == BLACK)
#     for i in range(len(my_index[0])):
#         if visited[my_index[0][i], my_index[1][i], 0]:
#             continue
#         liberties, connected_pieces = find_connected(board, my_index[1][i], my_index[0][i], move_color, mode=LIBERTY)
#         for cx, cy in connected_pieces:
#             visited[cy][cx][0] = True
#         for cx, cy in liberties:
#             liberty_board[cy][cx][0] = 1
#     for i in range(len(opponent_index[0])):
#         if visited[opponent_index[0][i], opponent_index[1][i], 1]:
#             continue
#         liberties, connected_pieces = find_connected(board, opponent_index[1][i], opponent_index[0][i], move_color*-1, mode=LIBERTY)
#         for cx, cy in connected_pieces:
#             visited[cy][cx][1] = True
#         for cx, cy in liberties:
#             liberty_board[cy][cx][1] = 1

#     return liberty_board

def liberty(board, pos_x, pos_y, move_color):
    """
    對指定位置的棋子計算氣，每一步只需計算落子的位置，與周圍的棋子
    channel 0: 棋盤 channel 3: 黑子的氣, channel 4: 白子的氣
    """
    dx = [0, 0, 1, -1]
    dy = [1, -1, 0, 0]
    board[pos_y][pos_x][3] = 0
    board[pos_y][pos_x][4] = 0
    for i in range(4):
        x = pos_x + dx[i]
        y = pos_y + dy[i]
        if x < LEFT_EDGE or x > RIGHT_EDGE or y < TOP_EDGE or y > BOTTOM_EDGE:
            continue
        if board[y][x][0] == 0:
            if move_color == BLACK:
                board[y][x][3] = 1
            else:
                board[y][x][4] = 1
    return board

class Board:
    """
    可視化棋盤，來驗證演算法是否正確
    """
    def __init__(self, master, size=19):
        self.master = master
        self.size = size
        self.canvas = tk.Canvas(self.master, width=600, height=600)
        self.canvas.pack()
        self.draw_board()
        
    def draw_board(self):
        for i in range(self.size):
            x0, y0 = self._get_coords(i, 0)
            x1, y1 = self._get_coords(i, self.size-1)
            self.canvas.create_line(x0, y0, x1, y1)
            
            x0, y0 = self._get_coords(0, i)
            x1, y1 = self._get_coords(self.size-1, i)
            self.canvas.create_line(x0, y0, x1, y1)
            
    def draw_stones(self, board, pos_x, pos_y):
        self.canvas.delete('stone')
        # print(liberty_board.shape)
        for j in range(self.size):
            for i in range(self.size):
                x, y = self._get_coords(i, j)
                if board[j][i][0] != 0:
                    # x, y = self._get_coords(i, j)
                    # liberty_board[j, i, 1]
                    # print(i, j)
                    if board[j][i][0] == 1:                        
                        self.canvas.create_oval(x-15, y-15, x+15, y+15, fill='black', tags='stone')
                        # self.canvas.create_text(x, y, text=str(liberties), tags='stone_text', fill='white')
                    else:
                        self.canvas.create_oval(x-15, y-15, x+15, y+15, fill='white', tags='stone')
                        # self.canvas.create_text(x, y, text=str(liberties), tags='stone_text', fill='black')
                    if i == pos_x and j == pos_y:
                        self.canvas.create_oval(x-4, y-4, x+4, y+4, fill='red', tags='stone')
                    # print(liberty_board[j][i][1])
                if board[j][i][3] == 1:
                    # print(i, j)
                    self.canvas.create_rectangle(x-8, y-8, x+8, y+8, fill='green', tags='stone')
                if board[j][i][4] == 1:
                    # print(i, j)
                    self.canvas.create_rectangle(x-4, y-4, x+4, y+4, fill='blue', tags='stone')

    def _get_coords(self, i, j):
        x = (i+1) * 30
        y = (j+1) * 30
        return x, y



def test_find_connected():
    """
    測試提子演算法與氣的計算
    """
    chars = 'abcdefghijklmnopqrst'
    coordinates = {k:v for v,k in enumerate(chars)}
    color_map = {'B':1, 'W':-1}

    rank = "dan"
    df = open(f'./CSVs/{rank}_train.csv').read().splitlines()
    games = [i.split(',',2)[2] for i in df]

    for game in games[:5]:
        moves_list = game.split(',')
        move_len = len(moves_list)
        board = np.zeros((19,19, 6))
        root = tk.Tk()
        board_gui = Board(root)
        for move_idx in range(move_len):
            move = moves_list[move_idx]# move: 'B[pq]'
            color = color_map[move[0]]
            pos_x = coordinates[move[2]]
            pos_y = coordinates[move[3]]
            board[pos_y][pos_x] = color
            board = capture_piece(board, pos_x, pos_y, color)
            board = liberty(board, pos_x, pos_y, color)
            # print(liberty_board[:, :, 1])
            board_gui.draw_stones(board, pos_x, pos_y)
            root.update()
            root.after(200)
    
    root.mainloop()

if __name__ == '__main__':
    test_find_connected()
