import numpy as np

'''return: shuffled data with indice'''
def paired_shuffle(dats, labels):
    indice = [idx for idx in range(len(dats))]

    np.random.shuffle(indice)
    shuffled_dats = []
    shuffled_labels = []
    for idx in indice:
        shuffled_dats.append(dats[idx])
        shuffled_labels.append(labels[idx])
    return shuffled_dats, shuffled_labels, indice


def queue_sort(queue):
    input_ex =[]
    while True:
        try:
            (i, dat) =queue.get_nowait()
            input_ex.append((i, dat))
        except:
            break
    input_ex =sorted(input_ex, key=lambda x: x[0])
    return [item[1] for item in input_ex]


def kfold(n_sample, n_split=5, shuffle=False):
    indice = [idx for idx in range(n_sample)]
    if shuffle:
        np.random.shuffle(indice)

    split_size = n_sample // n_split
    for ns in range(n_split):
        if (ns+1)*split_size > n_sample:
            idx_val = indice[ns*split_size:]
        else:
            idx_val = indice[ns*split_size: (ns+1)*split_size]
        idx_train = [idx for idx in filter(lambda x: x not in idx_val, indice)]

        yield idx_train, idx_val
