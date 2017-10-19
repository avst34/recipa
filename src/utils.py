import numpy as np

def lists_sum(lists):
    return sum(lists, [])

def stack_2d_arrays_pad_rows(arrays):
    max_length = max([x.shape[0] for x in arrays])
    return np.stack([np.pad(x, [(0, max_length - x.shape[0]), (0,0)], 'constant', constant_values=0) for x in arrays])