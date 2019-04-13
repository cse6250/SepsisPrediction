import os
import pickle
import pandas as pd

PATH_TRAIN = "./../../data/sepsis/train/train_sample_cleaned_pivoted_vital.csv"
PATH_VALIDATION = "./../../data/sepsis/validation/valid_sample_cleaned_pivoted_vital.csv"
PATH_TEST = "./../../data/sepsis/test/test_sample_cleaned_pivoted_vital.csv"
PATH_OUTPUT = "./../../data/sepsis/processed_long_term/"


def create_dataset(path, time_step=3):
    """
    :param path: path to the directory contains raw files.
    :param timestep: time step for the sample
    :return: List(patient IDs), List(labels), time sequence data as a List of List of List.
    """
    seqs = []
    labels = []
    df = pd.read_csv(path)
    grouped_df = df.groupby('icustay_id')
    for name, group in grouped_df:
        data = group.iloc[:, 2:-1]
        label = group.iloc[:, -1]
        record_seqs = []
        label_seqs = []
        for i in range(0, data.shape[0], time_step):
            if i + time_step > data.shape[0]:
                break
            record_seqs.append(data.iloc[i:i+time_step].mean(0).tolist())
            label_seqs.append(label.iloc[i:i+time_step].max(0))
        
        seqs.append(record_seqs)
        labels.append(max(label_seqs))

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
