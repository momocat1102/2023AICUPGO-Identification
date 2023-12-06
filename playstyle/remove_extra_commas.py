df = open('./CSVs/play_style_test_public.csv').read().splitlines()
id = [i.split(',',1)[0] for i in df]
games = [i.split(',',1)[1].replace(',,', '') for i in df]

with open('./CSVs/play_style_test_public_nocommas.csv', 'w') as f:
    for i in range(len(id)):
        if games[i][-1] == ',':
            f.write(id[i] + ',' + games[i][:-1] + '\n')
        else:
            f.write(id[i] + ',' + games[i] + '\n')

