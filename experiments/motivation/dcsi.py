import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree


def dcsiscore(data, partition, min_pts=5):
    clusters = partition
    for i in range(len(clusters)):
        if np.sum(partition == clusters[i]) == 1:
            partition[partition == clusters[i]] = -1
            clusters[i] = -1
    # all clusters except for -1
    clusters = np.setdiff1d(clusters, -1)
    # if no clusters left or just one cluster left return 0
    if len(clusters) == 0 or len(clusters) == 1:
        return 0
    # exclude noise points from dataset
    data = data[partition != -1, :]
    # calculate squared euclidean distance
    dist = squareform(pdist(data)) ** 2

    # original labelling
    poriginal = partition
    # exclude noise points from labeling
    partition = partition[partition != -1]
    cluster_labels = np.unique(partition)
    n_clusters = len(cluster_labels)
    dcsi = 0
    MST = {}
    CORE_PTS = {}
    for i in range(0, n_clusters):
        # indices of objects in cluster i
        objects_cl = np.where(partition == clusters[i])[0]
        # distance in the cluster
        dist_i = dist[np.ix_(objects_cl, objects_cl)]
        epsilon = calculate_epsilon(dist_i, 2 * min_pts)
        CORE_PTS[cluster_labels[i]] = core_points(dist_i, epsilon, min_pts)
        if len(CORE_PTS[cluster_labels[i]]) == 0:
            return -1
        dist_i = dist_i[np.ix_(CORE_PTS[cluster_labels[i]], CORE_PTS[cluster_labels[i]])]
        MST[cluster_labels[i]] = minimal_spanning_tree(dist_i)

    for i in range(0, n_clusters - 1):
        for j in range(i + 1, n_clusters):
            part = pairwise_dcsi(MST, CORE_PTS, data, partition, cluster_labels[i], cluster_labels[j])

            dcsi = dcsi + part
    dcsi = (2 / (n_clusters * (n_clusters - 1))) * dcsi

    return dcsi


def calculate_epsilon(dist_i, k):
    distances = []
    for i in range(0, dist_i.shape[0]):
        dists = np.unique(dist_i[i])
        if k >= len(dists):
            distances.append(dists[-1])
        else:
            distances.append(dists[k])
    epsilon = np.median(distances)
    return epsilon


def core_points(dist, epsilon, min_pts):
    neighborhoods = []
    for i in range(len(dist)):
        row = []
        for j in range(len(dist)):
            if i != j:
                if dist[i, j] <= epsilon:
                    row.append(dist[i, j])
        neighborhoods.append(row)
    core_pts = [i for i in range(len(neighborhoods)) if len(neighborhoods[i]) > min_pts - 1]
    return core_pts


def pairwise_dcsi(MST, CORE_PTS, data, partition, i, j):
    sep_dcsi = pairwise_separation(CORE_PTS, data, partition, i, j)
    conn_dcsi = pairwise_connectedness(MST, i, j)
    q = sep_dcsi / conn_dcsi
    return q / (1 + q)


def pairwise_separation(CORE_POINTS, data, labels, i, j):
    # distances between core points in between C_i and C_j
    # subset to include internal nodes of cluster i only
    subset_i = data[labels == i, :]
    core_pts_i = CORE_POINTS[i]
    subset_i = subset_i[core_pts_i]
    # subset to include internal nodes of cluster j only
    subset_j = data[labels == j, :]
    core_pts_j = CORE_POINTS[j]
    subset_j = subset_j[core_pts_j]
    sep_dcsi_list = cdist(subset_i, subset_j, metric="euclidean") ** 2
    sep_dcsi = np.min(sep_dcsi_list)
    return sep_dcsi


def pairwise_connectedness(MST, i, j):
    conn_dcsi = max(cluster_conn(MST, i), cluster_conn(MST, j))
    return conn_dcsi


def cluster_conn(MST, i):
    """
    Conn_dcsi(C_i) = max d(x_i, x_j), (x_i, x_j) in V

    :param MST:
    :param i:
    :return:
    """
    # maximum edge weight of MST
    conn_dcsi = np.max(MST[i])
    return conn_dcsi


def minimal_spanning_tree(dist_i):
    # transform to array
    dist = np.array(dist_i)
    # calculate minimal spanning tree and extract adjacency matrix
    # this calculates Kruskal
    mst = minimum_spanning_tree(dist).toarray()
    # mst is upper triangular matrix, make it symmetric
    mst_temp = mst + mst.T
    return mst_temp


"""#' Function to calculate self developed "density cluster separability index" (DCSI)
#' 
#' eps = median distance to minPts*2-th neighbor for each class
#' @param dist a distance matrix
#' @param labels a vector with labels
#' @param minPts minPts argument for core point definition
calc_DCSI <- function(dist, labels, minPts = 5){

  dist <- as.matrix(dist)

  # compute core points: 
  # calculate eps for each class = median distance to minPts*2-th nearest neighbor
  # calculate distance to minPts-th nearest neighbor among points of same class
  ind_corePoints <- c()
  for(i in unique(labels)){

    ind_i <- which(labels == i) 

    dist_i <- dist[ind_i, ind_i]

    knn_graph_eps_i <- cccd::nng(dx = dist_i, k = minPts*2)
    knn_matrix_eps_i <- as.matrix(igraph::as_adjacency_matrix(knn_graph_eps_i))
    knn_weights_eps_i <- matrixcalc::hadamard.prod(knn_matrix_eps_i, dist_i) # add distances to knn-graph

    dist_kth_neighbor_eps_i <- apply(knn_weights_eps_i, 1, max) # maximum value of every row = distance to minPts*2-th neighbor
    eps_i <- median(dist_kth_neighbor_eps_i)


    # calculate core points
    knn_graph_i <- cccd::nng(dx = dist_i, k = minPts)
    knn_matrix_i <- as.matrix(igraph::as_adjacency_matrix(knn_graph_i))
    knn_weights_i <- matrixcalc::hadamard.prod(knn_matrix_i, dist_i) # add distances to knn-graph

    dist_kth_neighbor_i <- apply(knn_weights_i, 1, max)

    ind_corePoints_i <- which(dist_kth_neighbor_i <= eps_i)

    ind_corePoints <- c(ind_corePoints, ind_i[ind_corePoints_i])

  }

  # from now on, consider only core points
  dist_core <- dist[ind_corePoints, ind_corePoints]
  labels_core <- labels[ind_corePoints]

  # for each cluster i: 
  # separation = minimum distance between a core point of i and a core point that is not in i
  # connectedness = maximum distance in a MST built of the core points of cluster i
  Sep_list <- list()
  Conn_list <- list()



  """
