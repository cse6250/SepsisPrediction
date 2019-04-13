import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, Dataset

def calculate_num_features(seqs):
    """
    :param seqs:
    :return: the calculated number of features
    """
    return len(seqs[0][0])

class VisitSequenceWithLabelDataset(Dataset):
    def __init__(self, seqs, labels):
        """
        Args:
            seqs (list): list of patients (list) of visits (list) of codes (int) that contains visit sequences
            labels (list): list of labels (int)
        """
        self.labels = labels
        matrix_seqs = []
        for record_seqs in seqs:
            matrix_seqs.append(np.matrix(record_seqs))
        self.seqs = matrix_seqs

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.seqs[index], self.labels[index]


def time_collate_fn(batch):
    """
    DataLoaderIter call - self.collate_fn([self.dataset[i] for i in indices])
    Thus, 'batch' is a list [(seq1, label1), (seq2, label2), ... , (seqN, labelN)]
    where N is minibatch size, seq is a FloatTensor, and label is a LongTensor
    
    :returns
        seqs (FloatTensor) - 3D of batch_size X length X num_features
        lengths (LongTensor) - 1D of batch_size
        labels (LongTensor) - 1D of batch_size
    """
    batch_seq, batch_label = zip(*batch)

    num_features = batch_seq[0].shape[1]
    seq_lengths = list(map(lambda record_tensor: record_tensor.shape[0], batch_seq))

    seqs = []
    labels = []

    for i in range(0, len(seq_lengths), 1):
        seqs.append(batch_seq[i].tolist())
        labels.append(batch_label[i])

    seqs_tensor = torch.FloatTensor(np.stack(seqs, axis=0))
    lengths_tensor = torch.LongTensor(seq_lengths)
    labels_tensor = torch.LongTensor(labels)

    return (seqs_tensor, lengths_tensor), labels_tensor