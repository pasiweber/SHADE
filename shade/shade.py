# License: BSD 3 clause

from __future__ import annotations

import numpy as np
import torch
import sys

from dcdist import DCTree, DCTree_Clusterer
from clustpy.deep.autoencoders._abstract_autoencoder import _AbstractAutoencoder
from clustpy.deep.autoencoders import FeedforwardAutoencoder
from clustpy.deep._data_utils import get_dataloader
from clustpy.deep._train_utils import get_trained_autoencoder
from clustpy.deep._utils import (
    detect_device,
    set_torch_seed,
    squared_euclidean_distance,
    encode_batchwise,
    run_initial_clustering,
)
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state
from tqdm import tqdm
from typing import Callable, Optional, Tuple

from clustpy.utils import plot_with_transformation
import matplotlib.pyplot as plt


sys.setrecursionlimit(1000000000)


class SHADE(BaseEstimator, ClusterMixin):
    """
    An autoencoder (AE) will be trained with the reconstruction loss and the d_dc loss function.
    Afterward, KMeans or HDBSCAN identifies the initial clusters.

    Parameters
    ----------
    batch_size : int
        Size of the data batches. (default: 256)
    n_epochs : int
        Number of n_epochs. (default: 50)
    embedding_size : int
        Size of the embedding within the autoencoder. (default: 10)
    autoencoder : torch.nn.Module
        The input autoencoder. If None a new Autoencoder will be created. (default: None)
    optimizer_params : dict
        Parameters of the optimizer for the clustering procedure.
        Can also include the learning rate. (default: {"lr": 1e-3})
    optimizer_class : torch.optim.Optimizer
        The optimizer class. (default: torch.optim.Adam)
    loss_fn : torch.nn.modules.loss._Loss
        Loss function for the reconstruction. (default: torch.nn.MSELoss())
    custom_dataloaders : tuple
        Tuple consisting of a trainloader (random order) at the first and
        a test loader (non-random order) at the second position.
        If None, the default dataloaders will be used. (default: None)
    random_state : np.random.RandomState
        Use a fixed random state to get a repeatable solution.
        Can also be of type int. (default: None)
    device : torch.device
        If device is None then it will set to cuda if it is available. (default: None)

    Attributes
    ----------
    labels_ : np.ndarray
        The final labels (obtained by a final KMeans execution)
    autoencoder : torch.nn.Module
        The final autoencoder

    Examples
    --------
    >>> from clustpy.data import create_subspace_data
    >>> data, labels = create_subspace_data(1500, subspace_features=(3, 50), random_state=1)
    >>> shade = SHADE()
    >>> shade.fit(data)

    References
    ----------
    //TODO: Fill in when published.
    """

    batch_size: int
    autoencoder: Optional[torch.nn.Module]
    min_points: int
    use_complete_dc_tree: bool
    use_matrix_dc_distance: bool
    increase_inter_cluster_distance: bool
    pretrain_epochs: int
    pretrain_optimizer_params: dict
    clustering_epochs: int
    clustering_optimizer_params: dict
    embedding_size: int
    optimizer_params: dict
    optimizer_class: torch.optim.Optimizer
    loss_fn: torch.nn.modules.loss._Loss
    custom_dataloaders = Optional[Tuple[Callable, Callable]]
    standardize: bool
    standardize_axis: int
    random_state: np.random.RandomState
    device: torch.device
    cluster_algorithm: ClusterMixin
    cluster_algorithm_params: dict
    degree_of_reconstruction: float
    degree_of_density_preservation: float

    n_clusters: Optional[int]
    labels_: np.ndarray

    def __init__(
        self,
        batch_size: int = 500,
        autoencoder: Optional[torch.nn.Module] = None,
        min_points: int = 5,
        use_complete_dc_tree: bool = False,
        use_matrix_dc_distance: bool = True,
        increase_inter_cluster_distance: bool = False,
        pretrain_epochs: int = 0,
        pretrain_optimizer_params: dict = {"lr": 1e-3},
        clustering_epochs: int = 100,
        clustering_optimizer_params: dict = {"lr": 1e-3},
        embedding_size: int = 10,
        optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
        loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(),
        custom_dataloaders: Optional[Tuple[Callable, Callable]] = None,
        standardize: bool = True,
        standardize_axis: int = 0,
        random_state: Optional[np.random.RandomState] = None,
        device: Optional[torch.device] = None,
        n_clusters: Optional[int] = None,
        cluster_algorithm: Optional[ClusterMixin] = DCTree_Clusterer,
        cluster_algorithm_params: dict = {},
        degree_of_reconstruction: float = 1.0,
        degree_of_density_preservation: float = 1.0,
    ):
        self.batch_size = batch_size
        self.min_points = min_points
        self.use_complete_dc_tree = use_complete_dc_tree
        self.dc_tree = None
        self.use_matrix_dc_distance = use_matrix_dc_distance
        self.increase_inter_cluster_distance = increase_inter_cluster_distance
        self.pretrain_epochs = pretrain_epochs
        self.pretrain_optimizer_params = pretrain_optimizer_params
        self.clustering_epochs = clustering_epochs
        self.clustering_optimizer_params = clustering_optimizer_params
        self.embedding_size = embedding_size
        self.autoencoder = autoencoder
        self.optimizer_class = optimizer_class
        self.loss_fn = loss_fn
        self.custom_dataloaders = custom_dataloaders
        self.standardize = standardize
        self.standardize_axis = standardize_axis
        self.random_state = check_random_state(random_state)
        set_torch_seed(self.random_state)
        self.device = detect_device(device)
        self.n_clusters = n_clusters
        self.cluster_algorithm = cluster_algorithm
        self.cluster_algorithm_params = cluster_algorithm_params
        self.degree_of_reconstruction = degree_of_reconstruction
        self.degree_of_density_preservation = degree_of_density_preservation

    def fit(self, X, y=None) -> SHADE:
        """
        Cluster the input dataset with the SHADE algorithm.
        The resulting cluster labels will be stored in the `labels_` attribute.

        Parameters
        ----------
        X : np.ndarray
            The given data set.
        y : np.ndarray
            The labels. (can be ignored)
        dc_distances : Optional[Union[np.ndarray, DCTree, SampleDCTree]]
            dc_distances of X.

        Returns
        -------
        self : SHADE
            This instance of the SHADE algorithm.
        """

        # Create Dataloader
        if self.custom_dataloaders is None:
            if self.standardize:
                Z = _standardize(X, self.standardize_axis)
            else:
                Z = X
            trainloader = get_dataloader(
                Z,
                self.batch_size,
                drop_last=True,
                shuffle=True,
            )
            testloader = get_dataloader(
                Z,
                self.batch_size,
                drop_last=False,
                shuffle=False,
            )
        else:
            trainloader, testloader = self.custom_dataloaders
            if trainloader.batch_size != self.batch_size:
                self.batch_size = trainloader.batch_size

        # Create dc_tree
        if self.dc_tree is None and self.use_complete_dc_tree:
            print("Build DCTree of the complete dataset")
            self.dc_tree = DCTree(X, min_points=self.min_points)

        # Create and pretrain Autoencoder
        if self.autoencoder is None:
            architecture = [X.shape[1], 512, 256, 128, self.embedding_size]
            self.autoencoder = FeedforwardAutoencoder(architecture)
            self.autoencoder = self.autoencoder.to(self.device)

        if not self.autoencoder.fitted:
            self.autoencoder = get_trained_autoencoder(
                trainloader,
                self.pretrain_optimizer_params,
                self.pretrain_epochs,
                self.device,
                self.optimizer_class,
                self.loss_fn,
                self.embedding_size,
                self.autoencoder,
            )
            self.autoencoder.fitted = False

        # Setup SHADE Module
        self.ddc_module = _SHADE_Module(
            autoencoder=self.autoencoder,
            n_epochs=self.clustering_epochs,
            min_points=self.min_points,
            dc_tree=self.dc_tree,
            use_matrix_dc_distance=self.use_matrix_dc_distance,
            increase_inter_cluster_distance=self.increase_inter_cluster_distance,
            degree_of_reconstruction=self.degree_of_reconstruction,
            degree_of_density_preservation=self.degree_of_density_preservation,
            optimizer_class=self.optimizer_class,
            optimizer_params=self.clustering_optimizer_params,
            device=self.device,
        )
        if not self.autoencoder.fitted:
            print("Start training with clustering loss.")
            self.ddc_module.fit(
                X,
                trainloader=trainloader,
                loss_fn=self.loss_fn,
                testloader=testloader,
            )
            self.autoencoder.fitted = True

        embedding = encode_batchwise(testloader, self.autoencoder, self.device)

        (
            self.n_clusters,
            self.labels_,
            self.cluster_centers_,
            _,
        ) = run_initial_clustering(
            X=embedding,
            n_clusters=self.n_clusters,
            clustering_class=self.cluster_algorithm,
            clustering_params=self.cluster_algorithm_params,
            random_state=self.random_state,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the labels of the input data.

        Parameters
        ----------
        X : np.ndarray
            The input data.
        n_clusters : int
            Number of clusters. Can be None if a corresponding initial_clustering_class is given,
            e.g. DBSCAN / HDBSCAN.
        cluster_algorithm : Clusterer, optional
            Clustering algorithm which should be used on the learned embedding. (default: KMeans)
        cluster_params : dict
            Clustering parameters for `cluster_algorithm`.

        Returns
        -------
        predicted_labels : np.ndarray
            The predicted labels.
        """

        if self.standardize:
            Z = _standardize(X, self.standardize_axis)
        else:
            Z = X
        dataloader = get_dataloader(
            Z,
            self.batch_size,
            drop_last=False,
            shuffle=False,
        )

        embedding = encode_batchwise(dataloader, self.autoencoder, self.device)

        (
            self.n_clusters,
            self.labels_,
            self.cluster_centers_,
            _,
        ) = run_initial_clustering(
            X=embedding,
            n_clusters=self.n_clusters,
            clustering_class=self.cluster_algorithm,
            clustering_params=self.cluster_algorithm_params,
            random_state=self.random_state,
        )
        return self.labels_

    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        Embedds the input data with the learned SHADE Autoencoder.

        Parameters
        ----------
        X : np.ndarray
            The input data.

        Returns
        -------
        np.ndarray
            Embedded input data.
        """

        if self.standardize:
            Z = _standardize(X, self.standardize_axis)
        else:
            Z = X
        dataloader = get_dataloader(
            Z,
            self.batch_size,
            drop_last=False,
            shuffle=False,
        )

        embedding = encode_batchwise(dataloader, self.autoencoder, self.device)
        return embedding


class _SHADE_Module(_AbstractAutoencoder):
    """
    The SHADE Autoencoder.
    """

    autoencoder: torch.nn.Module
    dc_tree: Optional[DCTree]
    use_matrix_dc_distance: bool
    n_epochs: int
    min_points: int
    increase_inter_cluster_distance: bool
    degree_of_reconstruction: float
    degree_of_density_preservation: float
    optimizer: torch.optim.Optimizer
    device: torch.device

    def __init__(
        self,
        autoencoder: torch.nn.Module,
        n_epochs=100,
        min_points: int = 5,
        dc_tree: Optional[DCTree] = None,
        use_matrix_dc_distance: bool = True,
        increase_inter_cluster_distance: bool = False,
        degree_of_reconstruction: float = 1.0,
        degree_of_density_preservation: float = 1.0,
        optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
        optimizer_params: dict = {},
        device: Optional[torch.device] = None,
        l=None,
    ):
        super().__init__()

        self.autoencoder = autoencoder
        self.n_epochs = n_epochs
        self.min_points = min_points
        self.dc_tree = dc_tree
        self.use_matrix_dc_distance = use_matrix_dc_distance
        self.increase_inter_cluster_distance = increase_inter_cluster_distance
        self.degree_of_reconstruction = degree_of_reconstruction
        self.degree_of_density_preservation = degree_of_density_preservation
        self.optimizer = optimizer_class(list(autoencoder.parameters()), **optimizer_params)
        self.device = detect_device(device)
        self.l = l

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.encode(x)

    def decode(self, embedded: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.decode(embedded)

    def fit(
        self,
        X,
        trainloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.modules.loss._Loss,
        testloader,
    ) -> _SHADE_Module:
        """
        Trains the autoencoder in place.

        Parameters
        ----------
        trainloader : torch.utils.data.DataLoader
            Dataloader to be used for training.
        dc_distances : Union[np.ndarray, DCTree, SampleDCTree]
            dc_distances of the data of the dataloader.
        n_epochs : int
            Number of epochs for the clustering procedure.
        optimizer : torch.optim.Optimizer
            The optimizer for training.
        loss_fn : torch.nn.modules.loss._Loss
            Loss function for the reconstruction.
        device : torch.device
            Device to be trained on.

        Returns
        -------
        self : _SHADE_Module
            This instance of the _SHADE_Module.
        """

        if self.dc_tree is not None and self.use_matrix_dc_distance:
            self.matrix_dc_distance = self.dc_tree.dc_distances()

        self.train()
        for epoch_i in tqdm(range(self.n_epochs), file=sys.stdout, desc="Epoch", disable=True):
            loss_rec_sum = []
            loss_dens_sum = []
            # Update Network
            for batch in trainloader:
                if len(batch[0]) <= self.min_points:
                    continue

                loss_rec, loss_dens = self._loss(X, batch, loss_fn)
                loss = (
                    self.degree_of_reconstruction * loss_rec
                    + self.degree_of_density_preservation * loss_dens
                )
                loss_rec_sum.append(loss_rec.cpu().detach().numpy())
                loss_dens_sum.append(loss_dens.cpu().detach().numpy())
                # Backward pass - update weights
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            # print(
            #     epoch_i,
            #     np.array(loss_rec_sum).mean(),
            #     np.array(loss_dens_sum).mean(),
            # )
            # emb = encode_batchwise(testloader, self.autoencoder, self.device)
            # plot_with_transformation(emb, scattersize=1, show_plot=False)
            # dc_tree = DCTree(emb)
            # plot_mst(emb, dc_tree)
            plt.show()

        self.autoencoder.eval()
        self.eval()
        return self

    def _loss(
        self,
        X,
        batch: list,
        loss_fn: torch.nn.modules.loss._Loss,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the autoencoder reconstruction + d_dc loss.

        Parameters
        ----------
        batch : list
            The minibatch.
        loss_fn : torch.nn.modules.loss._Loss
            Loss function for the reconstruction.
        device : torch.device
            Device to be trained on.

        Returns
        -------loss
        loss : torch.Tensor
            The final SHADE loss.
        """

        # Reconstrucion
        batch_data = batch[1].to(self.device)
        emb_data = self.encode(batch_data)
        reconstructed = self.decode(emb_data)
        loss_rec = loss_fn(reconstructed, batch_data)

        # Density loss
        if self.dc_tree is None:
            # Batch-wise DCTree
            if self.increase_inter_cluster_distance:
                dc_clusterer = DCTree_Clusterer(
                    min_points=self.min_points,
                    increase_inter_cluster_distance=self.increase_inter_cluster_distance,
                )
                dc_clusterer.fit_predict(X[batch[0]])
                batch_dc_dists = torch.tensor(
                    dc_clusterer.dc_tree.dc_distances(), device=self.device
                )
            else:
                batch_dc_dists = torch.tensor(
                    DCTree(X[batch[0]], min_points=self.min_points).dc_distances(),
                    device=self.device,
                )
        else:
            # DCTree of all data points X
            if self.use_matrix_dc_distance:
                batch_dc_dists = torch.tensor(
                    self.matrix_dc_distance[np.ix_(batch[0], batch[0])], device=self.device
                )
            else:
                batch_dc_dists = torch.tensor(
                    self.dc_tree.dc_distances(batch[0], batch[0]), device=self.device
                )

        batch_eucl_dists = squared_euclidean_distance(emb_data, emb_data)
        loss_dens = (batch_eucl_dists - batch_dc_dists).pow(2).mean()

        return loss_rec, loss_dens


def _standardize(X: np.ndarray, axis: Optional[int] = 0):
    mean = np.mean(X, axis=axis)
    std = np.std(X, axis=axis)
    if isinstance(std, np.ndarray):
        std[std == 0] = 1
    elif std == 0:
        std = 1
    return (X - mean) / std
