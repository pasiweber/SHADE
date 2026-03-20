import numpy as np
import os

from clustpy.data import *
from enum import Enum


CURRENT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
DATASETS_FOLDER = f"{CURRENT_DIRECTORY}"


### Datasets ###

class Datasets(Enum):
    # Tabular data
    Synth_low = "Synth_low"
    Synth_high = "Synth_high"
    HAR = "HAR"
    letterrec = "letterrec."
    htru2 = "htru2"
    Mice = "Mice"
    TCGA = "TCGA"
    Pendigits = "Pendigits"
    # Video data
    Weizmann = "Weizmann"
    Keck = "Keck"
    # Image data
    COIL20 = "COIL20"
    COIL100 = "COIL100"
    cmu_faces = "cmu_faces"
    # MNIST data
    Optdigits = "Optdigits"
    USPS = "USPS"
    MNIST = "MNIST"
    FMNIST = "FMNIST"
    KMNIST = "KMNIST"

    @property
    def id(self):
        return super(Datasets, self).name

    @property
    def name(self):
        return super(Datasets, self).value

    @property
    def data(self):
        """Returns (X, l), with X standardized and caches (X, l)"""
        return load_and_cache_standardized_dataset(self)

    @property
    def original_data(self):
        """Returns (X, l)"""
        return load_original_dataset(self)

    @property
    def standardized_data(self):
        """Returns (X, l), with X standardized"""
        X, l = load_original_dataset(self)
        return standardize_dataset(self, X, l)


def load_np_dataset(path, from_csv=False):
    if from_csv:
        X = np.genfromtxt(path + "_data.csv", delimiter=',')
    else:
        X = np.load(path + "_data.npy", allow_pickle=True)
    l = np.load(path + "_labels.npy", allow_pickle=True)
    X = X.reshape((len(X), -1))
    return X, l

def load_syn(path):
    D = np.load(path)
    X, l = D[:, :-1], D[:, -1]
    return X, l

def video_labels(X, l):
    l = l[:, 1] * len(np.unique(l[:, 0])) + l[:, 0]
    return X, l

def apply_label(X, l, nr):
    l = l[:, nr]
    return X, l

def load_from_clustpy(dataset):
    return dataset.data, dataset.target

def load_original_dataset(dataset_id):
    match dataset_id:
        # Tabular data
        case Datasets.Synth_low: return load_syn(f"{DATASETS_FOLDER}/low_data_100.npy")
        case Datasets.Synth_high: return load_syn(f"{DATASETS_FOLDER}/high_data_100.npy")
        case Datasets.HAR: return load_from_clustpy(load_har())
        case Datasets.letterrec: return load_from_clustpy(load_letterrecognition())
        case Datasets.htru2: return load_from_clustpy(load_htru2())
        case Datasets.Mice: return load_from_clustpy(load_mice_protein())
        case Datasets.TCGA: return load_np_dataset(f"{DATASETS_FOLDER}/TCGA_PANCAN_HiSeq", from_csv=True)
        case Datasets.Pendigits: return load_from_clustpy(load_pendigits())
        # Video data
        case Datasets.Weizmann: return video_labels(*load_from_clustpy(load_video_weizmann()))
        case Datasets.Keck: return video_labels(*load_from_clustpy(load_video_keck_gesture()))
        # Image data
        case Datasets.COIL20: return load_from_clustpy(load_coil20())
        case Datasets.COIL100: return load_from_clustpy(load_coil100())
        case Datasets.cmu_faces: return apply_label(*load_from_clustpy(load_cmu_faces()), 0)
        # MNIST data
        case Datasets.Optdigits: return load_from_clustpy(load_optdigits())
        case Datasets.USPS: return load_from_clustpy(load_usps())
        case Datasets.MNIST: return load_from_clustpy(load_mnist())
        case Datasets.FMNIST: return load_from_clustpy(load_fmnist())
        case Datasets.KMNIST: return load_from_clustpy(load_kmnist())
        case _:
            raise AttributeError


def standardize(X, l, axis=None):
    std = np.std(X, axis=axis)
    if axis != None:
        X = X[:, std != 0]  # Remove features which are identical over all samples
    mean = np.mean(X, axis=axis)
    X = (X - mean) / std[std != 0]
    return X, l

def standardize_dataset(dataset_id, X, l):
    match dataset_id:
        case dataset if dataset in [
            # Tabular data
            Datasets.Synth_low, 
            Datasets.Synth_high,
            Datasets.HAR,
            Datasets.letterrec,
            Datasets.htru2,
            Datasets.Mice,
            Datasets.TCGA,
            Datasets.Pendigits,
        ]:
            return standardize(X, l, axis=0)
        case dataset if dataset in [
            # Video data
            Datasets.Weizmann,
            Datasets.Keck,
            # Image data
            Datasets.COIL20,
            Datasets.COIL100,
            Datasets.cmu_faces,
            # MNIST data
            Datasets.Optdigits,
            Datasets.USPS,
            Datasets.MNIST,
            Datasets.FMNIST,
            Datasets.KMNIST,
        ]: 
            return standardize(X, l, axis=None)
        case _:
            raise AttributeError


def load_and_cache_standardized_dataset(dataset_id):
    if not os.path.exists(f"{DATASETS_FOLDER}/.cache/"):
        os.makedirs(f"{DATASETS_FOLDER}/.cache/") 
    cache_path = f"{DATASETS_FOLDER}/.cache/{dataset_id}"
    if os.path.exists(f"{cache_path}_data.npy") and os.path.exists(f"{cache_path}_labels.npy"):
        return load_np_dataset(cache_path)
    else:
        X, l = load_original_dataset(dataset_id)
        X, l = standardize_dataset(dataset_id, X, l)
        np.save(f"{cache_path}_data.npy", X)
        np.save(f"{cache_path}_labels.npy", l)
        return X, l
