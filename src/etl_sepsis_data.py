import os
import pickle
import pandas as pd

PATH_TRAIN = "../data/sepsis/train/"
PATH_VALIDATION = "../data/sepsis/validation/"
PATH_TEST = "../data/sepsis/test/"
PATH_OUTPUT = "../data/sepsis/processed/"


def create_dataset(path, time_step=3):
    """
    :param path: path to the directory contains raw files.
    :param timestep: time step for the sample
    :return: List(patient IDs), List(labels), time sequence data as a List of List of List.
    """
    file_list =  os.listdir(path)
    seqs = []
    labels = []
    for file in sorted(file_list):
        df = pd.read_csv(os.path.join(path, file), sep='|')
        df.fillna(0, inplace=True)
        data = df.iloc[:, :-3]
        target = df.iloc[:, -1]
        record_seqs = []
        target_seqs = []
        for i in range(1, data.shape[0]+1, time_step):
            if i+time_step > data.shape[0] + 1:
                break
            record_seqs.append(data.iloc[i:i+time_step].mean(0).tolist())
            target_seqs.append(target.iloc[i:i+time_step].max(0))
        seqs.append(record_seqs)
        labels.append(target_seqs)

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
