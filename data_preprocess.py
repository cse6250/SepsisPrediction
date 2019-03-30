import os
import pandas as pd

PATH_TRAINING = 'training/'

def process():
    """
    NaN value is kept in the list
    """

    file_list = os.listdir(PATH_TRAINING)
    seqs = []
    labels = []
    for file in sorted(file_list):
        records = []
        df = pd.read_csv(os.path.join(PATH_TRAINING, file), sep='|')
        y_df = df.iloc[:, -1]
        labels.append(y_df.tolist())
        df.drop(columns=['SepsisLabel'], inplace=True)
        for _, row in df.sort_values('ICULOS').iterrows():
            statistics = []
            for val in row:
                statistics.append(val)
            records.append(statistics)
        seqs.append(records)
    return seqs, labels

if __name__ == '__main__':
    seqs, labels = process()