
def concate_csv(csvs, result_csv_path):
    with open(result_csv_path, 'w') as f:
        for csv in csvs:
            for line in open(csv).read().splitlines():
                f.write(line + '\n')

def main():
    # concate public and private test csvs
    dan_predict_csvs = ['./29_Public Testing Dataset_v2/dan_test_public.csv', './29_Private Testing Dataset_v2/dan_test_private.csv']
    concate_csv(dan_predict_csvs, './test_dataset/dan_test.csv')
    kyu_predict_csvs = ['./29_Public Testing Dataset_v2/kyu_test_public.csv', './29_Private Testing Dataset_v2/kyu_test_private.csv']
    concate_csv(kyu_predict_csvs, './test_dataset/kyu_test.csv')

if __name__ == '__main__':
    main()