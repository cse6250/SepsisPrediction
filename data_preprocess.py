import os
import pandas as pd

PATH_TRAINING = 'training/'

def process():
    file_list = os.listdir(PATH_TRAINING)
    seqs = []
    for file in sorted(file_list):
        records = []
        df = pd.read_csv(os.path.join(PATH_TRAINING, file), sep='|')
        for _, row in df.sort_values('ICULOS').iterrows():
            statistics = []
            for val in row:
                statistics.append(val)
            records.append(statistics)
        seqs.append(records)
    return seqs

if __name__ == '__main__':
    seqs = process()