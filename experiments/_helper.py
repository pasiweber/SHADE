import numpy as np
from clustpy.data import load_mice_protein
import pandas as pd


def load_np_dataset(path):
    X, l = (
        np.load(path + "_data.npy", allow_pickle=True),
        np.load(path + "_labels.npy", allow_pickle=True),
    )
    X = X.reshape((len(X), -1))
    return lambda *args, **kwargs: (X, l)


def video_labels(load_fn):
    X, l = load_fn(normalize_channels=True)
    l = l[:, 1] * len(np.unique(l[:, 0])) + l[:, 0]
    return lambda *args, **kwargs: (X, l)


def mice_behavior():
    X, l = load_mice_protein(return_additional_labels=True)
    return lambda *args, **kwargs: (X, l[:, 1])


def load_syn(path):
    D = np.load(path)
    X, l = D[:, :-1], D[:, -1]
    return lambda *args, **kwargs: (X, l)


def apply_label(func, nr, *args):
    X, l = func(*args)
    l = l[:, nr]
    return lambda *args, **kwargs: (X, l)


def standardize(func, axis=None, *args):
    X, l = func(*args)
    mean = np.mean(X, axis=axis)
    std = np.std(X, axis=axis)
    if isinstance(std, np.ndarray):
        std[std == 0] = 1
    elif std == 0:
        std = 1
    X = (X - mean) / std
    return lambda *args, **kwargs: (X, l)
