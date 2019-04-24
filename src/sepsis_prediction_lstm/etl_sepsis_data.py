import os
import pickle
import pandas as pd
from datetime import datetime
from datetime import timedelta

PATH_TRAIN = "./../../data/sepsis/train/train_sample_cleaned_pivoted_vital.csv"
PATH_VALIDATION = "./../../data/sepsis/validation/valid_sample_cleaned_pivoted_vital.csv"
PATH_TEST = "./../../data/sepsis/test/test_sample_cleaned_pivoted_vital.csv"
PATH_OUTPUT = "./../../data/sepsis/processed_long_term/"

def create_dataset(path, observation_window=12, prediction_window=3):
    """
    :param path: path to the directory contains raw files.
    :param observation window: time interval we will use to identify relavant events
    :param prediction window: a fixed time interval that is to be used to make the prediction
    :return: List(pivot vital records), List(labels), time sequence data as a List of List of List.
    """
    seqs = []
    labels = []
    count_sepsis = 0
    count_sepsis_with_six = 0
    # load data from csv;
    df = pd.read_csv(path)

    # construct features
    grouped_df = df.groupby('icustay_id')
    for name, group in grouped_df: 
        # calculate the index_hour
        # for the patients who have the sepsis, index hour is #prediction_window hours prior to the onset time
        # for the patients who don't have the sepsis, index hour is the last event time 
        if group.iloc[-1,-1] == 1:
            index_hour = datetime.strptime(group.iloc[-1,1], '%Y-%m-%d %H:%M:%S') - timedelta(hours=prediction_window)
        else:
            index_hour = datetime.strptime(group.iloc[-1,1], '%Y-%m-%d %H:%M:%S')

        # get the date in observation window
        group['record_time'] = pd.to_datetime(group['record_time'])
        filterd_group = group[(group['record_time'] >= (index_hour - timedelta(hours=observation_window))) & (group['record_time'] <= index_hour)]
        # construct the records seqs and label seqs
        data = filterd_group.iloc[:, 2:-1]
        record_seqs = []
        for i in range(0, data.shape[0], 1):
            record_seqs.append(data.iloc[i].tolist())

        if group.iloc[-1, -1]:
            count_sepsis += 1
            if len(record_seqs) != 0:
                count_sepsis_with_six += 1

        if len(record_seqs) != 0:
            seqs.append(record_seqs)
            labels.append(group.iloc[-1, -1])
    
    print(count_sepsis, count_sepsis_with_six)
    return seqs, labels


def main():
    os.makedirs(PATH_OUTPUT, exist_ok=True)
    # Train set
    print("Construct train set")
    train_seqs, train_labels = create_dataset(PATH_TRAIN)

    pickle.dump(train_seqs, open(os.path.join(PATH_OUTPUT, "sepsis.seqs.train"), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(train_labels, open(os.path.join(PATH_OUTPUT, "sepsis.labels.train"), 'wb'), pickle.HIGHEST_PROTOCOL)

    # Validation set
    print("Construct validation set")
    train_seqs, train_labels = create_dataset(PATH_VALIDATION)

    pickle.dump(train_seqs, open(os.path.join(PATH_OUTPUT, "sepsis.seqs.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(train_labels, open(os.path.join(PATH_OUTPUT, "sepsis.labels.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)

    # Test set
    print("Construct test set")
    train_seqs, train_labels = create_dataset(PATH_TEST)

    pickle.dump(train_seqs, open(os.path.join(PATH_OUTPUT, "sepsis.seqs.test"), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(train_labels, open(os.path.join(PATH_OUTPUT, "sepsis.labels.test"), 'wb'), pickle.HIGHEST_PROTOCOL)

    print("Complete!")


if __name__ == '__main__':
    main()
