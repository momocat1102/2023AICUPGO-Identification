import numpy as np
import matplotlib.pyplot as plt

def main():
    # 生成balance.npy，用於平衡資料
    x = np.zeros((420, ))
    mutiple = np.zeros((420, ))
    power = "kyu"
    balence_num = 80000 # 65000/80000
    # 讀取檔案
    min_len = 420
    max_len = 0
    df = open(f'CSVs/{power}_train.csv').read().splitlines()
    for i in range(len(df)):
        game = df[i].split(',',2)[2]
        moves_list = game.split(',')
        move_len = len(moves_list)
        #x[move_len] += 1
        x[:move_len+1] += 1
        if move_len < min_len:
            min_len = move_len
        if max_len < move_len:
            max_len = move_len

    print(max_len, min_len)
    mutiple[:250] = balence_num/x[:250]
    mutiple[250:] = np.where(x[250:] == 0, 0, (balence_num/x[250:]))
    # 將大於2.5的mutiple設為2.5
    mutiple = np.where(mutiple > 2.5, 2.5, mutiple)
    mutiple = np.where(mutiple == 0, 2.5, mutiple)
    np.save(f"balance_{power}.npy", mutiple)
    print(min_len)
    # 算大約會產生多少資料
    count = 0
    for i in range(420):
        count += x[i] * mutiple[i]
    print(count//2)

    # 畫出質方圖, x: 步數, y: 次數
    plt.bar(np.arange(1, 421), x)
    plt.show()
    plt.bar(np.arange(1, 421), mutiple)
    plt.show()

if __name__ == "__main__":
    main()