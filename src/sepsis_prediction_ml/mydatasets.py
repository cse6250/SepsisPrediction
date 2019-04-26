import numpy as np

def construct_features(seqs):
    X = []
    for i in range(0, len(seqs), 1):
        arr = np.array(seqs[i])
        data_mean = np.mean(arr, axis=0)
        X.append(data_mean.tolist())
    
    return X