import numpy as np
import torch

import sys
import os
dirname = os.path.dirname(__file__)
path = os.path.join(dirname, '../../')
sys.path.append(path)

sys.setrecursionlimit(1500000000)

from clustpy.deep.neural_networks import FeedforwardAutoencoder
from clustpy.deep._utils import embedded_kmeans_prediction
from clustpy.utils import (
    EvaluationDataset,
    EvaluationAlgorithm,
    EvaluationMetric,
    evaluate_multiple_datasets,
)
from clustpy.metrics import (
    unsupervised_clustering_accuracy as acc,
    information_theoretic_external_cluster_validity_measure as dom,
)
from sklearn.metrics import (
    normalized_mutual_info_score as nmi,
    adjusted_mutual_info_score as ami,
    adjusted_rand_score as ari,
)
from sklearn.utils import check_random_state

from clustpy.deep import (
    encode_batchwise,
    detect_device,
    get_dataloader,
    DEC,
    IDEC,
    DCN,
    DipEncoder,
    DDC,
    VaDE,
)
from clustpy.partition import SubKmeans
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from shade.shade import SHADE
from shade.shade_ship import SHADE_SHiP
from dcdist import DCTree_Clusterer

DOWNLOAD_PATH = None
SAVE_DIR = "experiments/ae_results/benchmark_ship/"

import os

os.environ["CLUSTPY_DEVICE"] = "cuda:0"


from experiments.datasets.datasets import Datasets

def get_X_l(dataset):
    X, l = dataset.standardized_data
    return X, np.array(l) + 1

def _get_dataset_loaders():
    datasets = [
        ("Synth_low", Datasets.Synth_low),
        ("Synth_high", Datasets.Synth_high),
        ("HAR", Datasets.HAR),
        ("letterrecognition", Datasets.letterrec),
        ("htru2", Datasets.htru2),
        ("Mice", Datasets.Mice),
        ("TCGA_HiSeq", Datasets.TCGA),
        ("Optdigits", Datasets.Optdigits),
        ("Pendigits", Datasets.Pendigits),
        ("USPS", Datasets.USPS),
        ("MNIST", Datasets.MNIST),
        ("FMNIST", Datasets.FMNIST),
        ("KMNIST", Datasets.KMNIST),
        ("Weizmann", Datasets.Weizmann),
        ("Keck", Datasets.Keck),
        ("COIL20", Datasets.COIL20),
        ("Coil100", Datasets.COIL100),
        ("cmu_faces", Datasets.cmu_faces),
    ]
    datasets = [(dataset_id, lambda dataset=dataset, *args, **kwargs: get_X_l(dataset)) for dataset_id, dataset in datasets]
    return datasets


def _get_evaluation_algorithms(
    n_clustering_epochs,
    embedding_size,
    batch_size,
    optimizer_class,
    loss_fn,
):
    evaluation_algorithms = [
        EvaluationAlgorithm(
            "SHADE_SHIP",
            SHADE_SHiP,
            {
                "n_clusters": None,
                "batch_size": batch_size,
                "pretrain_epochs": 0,
                "clustering_epochs": 100,
                "clustering_epochs": n_clustering_epochs,
                "optimizer_class": optimizer_class,
                "loss_fn": loss_fn,
                "embedding_size": embedding_size,
            },
        ),
    ]
    return evaluation_algorithms


def _get_evaluation_metrics():
    evaluation_metrics = [
        EvaluationMetric("NMI", nmi),
        EvaluationMetric("AMI", ami),
        EvaluationMetric("ARI", ari),
        EvaluationMetric("ACC", acc),
        EvaluationMetric("DOM", dom),
    ]
    return evaluation_metrics

def _get_evaluation_datasets_with_autoencoders(
    dataset_loaders,
    ae_layers,
    experiment_name,
    n_repetitions,
    batch_size,
    n_pretrain_epochs,
    optimizer_class,
    pretrain_optimizer_params,
    loss_fn,
    ae_class,
    other_ae_params,
    device,
):
    evaluation_datasets = []
    # Get autoencoders for DC algorithms
    for data_name_orig, data_loader in dataset_loaders:
        data_name_exp = data_name_orig + "_" + experiment_name
        eval_dataset = EvaluationDataset(
            data_name_exp,
            data_loader,
            train_test_split=False,
            preprocess_methods=[],
            preprocess_params=[],
        )
        evaluation_datasets.append(eval_dataset)
    return evaluation_datasets


def _experiment(
    experiment_name,
    ae_layers,
    embedding_size,
    n_repetitions,
    batch_size,
    n_pretrain_epochs,
    n_clustering_epochs,
    optimizer_class,
    pretrain_optimizer_params,
    loss_fn,
    ae_class,
    other_ae_params,
):
    ae_layers = ae_layers.copy()
    ae_layers.append(embedding_size)
    experiment_name = experiment_name + "_" + "_".join(str(x) for x in ae_layers)
    dataset_loaders = _get_dataset_loaders()
    device = detect_device()
    evaluation_datasets = _get_evaluation_datasets_with_autoencoders(
        dataset_loaders,
        ae_layers,
        experiment_name,
        n_repetitions,
        batch_size,
        n_pretrain_epochs,
        optimizer_class,
        pretrain_optimizer_params,
        loss_fn,
        ae_class,
        other_ae_params,
        device,
    )
    evaluation_algorithms = _get_evaluation_algorithms(
        n_clustering_epochs, embedding_size, batch_size, optimizer_class, loss_fn
    )
    evaluation_metrics = _get_evaluation_metrics()
    evaluate_multiple_datasets(
        evaluation_datasets,
        evaluation_algorithms,
        evaluation_metrics,
        n_repetitions,
        add_runtime=True,
        add_n_clusters=True,
        save_path=SAVE_DIR + experiment_name + "/Results/result.csv",
        save_intermediate_results=True,
        save_labels_path=SAVE_DIR + experiment_name + "/Labels/label.csv",
    )


def experiment_feedforward_512_256_128_10(
    n_repetitions=10,
    batch_size=500,
    n_pretrain_epochs=0,
    n_clustering_epochs=100,
    optimizer_class=torch.optim.Adam,
    pretrain_optimizer_params={"lr": 1e-3},
    loss_fn=torch.nn.MSELoss(),
    other_ae_params={},
):
    experiment_name = "ALL"
    embedding_size = 10
    ae_layers = [512, 256, 128]
    ae_class = FeedforwardAutoencoder
    _experiment(
        experiment_name,
        ae_layers,
        embedding_size,
        n_repetitions,
        batch_size,
        n_pretrain_epochs,
        n_clustering_epochs,
        optimizer_class,
        pretrain_optimizer_params,
        loss_fn,
        ae_class,
        other_ae_params,
    )


if __name__ == "__main__":
    experiment_feedforward_512_256_128_10()
