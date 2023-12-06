import numpy as np
EMPTY = 0
BLACK = 1
WHITE = -1

LEFT_EDGE = 0
RIGHT_EDGE = 18
TOP_EDGE = 0
BOTTOM_EDGE = 18

def find_connected(board, pos_x, pos_y, color):
    connected_pieces = []
    visited = np.zeros((19, 19),dtype=np.bool_)
    stack = []
    stack.append((pos_x, pos_y))
    visited[pos_y][pos_x] = True
    dx = [0, 0, 1, -1]
    dy = [1, -1, 0, 0]
    has_liberty = False
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
                has_liberty = True
                break 
            if board[new_y][new_x][0] == color:
                stack.append((new_x, new_y))
                visited[new_y][new_x] = True
                
    return has_liberty, connected_pieces

def capture_piece(board, pos_x, pos_y, move_color):
    dx = [0, 0, 1, -1]
    dy = [1, -1, 0, 0]
    # 設定黑子和白子的氣是在哪個channel
    black_channel = 1
    white_channel = 2
    for i in range(4):
        x = pos_x + dx[i]
        y = pos_y + dy[i]
        if x < LEFT_EDGE or x > RIGHT_EDGE or y < TOP_EDGE or y > BOTTOM_EDGE:
            continue
        
        if board[y][x][0] == -move_color:
            has_liberties, connected_pieces = find_connected(board, x, y, move_color*-1)

            if not has_liberties: # 沒氣
                for cx, cy in connected_pieces:
                    for i in range(4):
                        new_x = cx + dx[i]
                        new_y = cy + dy[i]
                        if new_x < LEFT_EDGE or new_x > RIGHT_EDGE or new_y < TOP_EDGE or new_y > BOTTOM_EDGE:
                            continue
                        if board[new_y][new_x][0] == move_color:
                            if move_color == BLACK:
                                board[cy][cx][black_channel] = 1
                                break
                            else:
                                board[cy][cx][white_channel] = 1
                                break
                    board[cy][cx][0] = EMPTY
    return board
def liberty(board, pos_x, pos_y, move_color):
    dx = [0, 0, 1, -1]
    dy = [1, -1, 0, 0]
    black_channel = 1 # 設定黑子的氣是在哪個channel
    white_channel = 2 # 設定白子的氣是在哪個channel
    
    board[pos_y][pos_x][black_channel] = 0
    board[pos_y][pos_x][white_channel] = 0
    
    for i in range(4):
        x = pos_x + dx[i]
        y = pos_y + dy[i]
        if x < LEFT_EDGE or x > RIGHT_EDGE or y < TOP_EDGE or y > BOTTOM_EDGE:
            continue
        if board[y][x][0] == EMPTY:
            if move_color == BLACK:
                board[y][x][black_channel] = 1
            else:
                board[y][x][white_channel] = 1
    
    return board
