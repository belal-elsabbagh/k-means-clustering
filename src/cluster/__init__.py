import numpy as np
import pandas as pd

from src.distance_functions import euclidean

VALUE_ERROR_MSG = "Either k or the initial cluster (init_c) must be provided"


def _init_c(df, k) -> dict:
    return {
        i + 1: row[1].tolist() for i, row in zip(range(k), df.sample(n=k).iterrows())
    }


def _assign_to_clusters(df, centroids, distance_fn):
    def get_row_cluster(row):
        d = {ind: distance_fn(row, c) for ind, c in centroids.items()}
        return min(d, key=d.get)

    return group_rows_by_cluster(
        {index: get_row_cluster(row.tolist()) for index, row in df.iterrows()}
    )


def group_rows_by_cluster(rows):
    clusters = {}
    for i, v in rows.items():
        clusters[v] = [i] if v not in clusters.keys() else clusters[v] + [i]
    return clusters


def _get_centers(df, clusters) -> dict:
    if clusters is None:
        return {}
    return {c_num: [np.mean(df[df.index.isin(rows)][f]) for f in df]for c_num, rows in clusters.items()}


def _init_params(df, k, distance_fn, init_c):
    if init_c is None and k is None:
        raise ValueError(VALUE_ERROR_MSG)
    return (
        euclidean if distance_fn is None else distance_fn,
        _init_c(df, k) if init_c is None else init_c,
    )


def _cluster(
    df: pd.DataFrame,
    k=None,
    max_epochs=None,
    distance_fn=None,
    init_c=None,
    verbose=False,
):
    def satisfied(centroids, centers, max_epochs, i):
        return centers == centroids or (max_epochs is not None and i >= max_epochs)

    distance_fn, init_centroids = _init_params(df, k, distance_fn, init_c)
    centers, centroids, clusters = init_centroids, None, None
    i = 0
    while not satisfied(centroids, centers, max_epochs, i):
        centroids = centers
        if verbose:
            print(f'Epoch {i+1}: {centroids}')
        clusters = _assign_to_clusters(df, centroids, distance_fn)
        centers = _get_centers(df, clusters)
        i += 1
    return clusters


class KMeans:
    def __init__(
        self, k=None, max_epochs=None, distance_fn=None, init_c=None, verbose=False
    ):
        self.k = k
        self.max_epochs = max_epochs
        self.distance_fn = distance_fn
        self.init_c = init_c
        self.verbose = verbose
        self.clusters = None

    def fit(self, df: pd.DataFrame):
        self.clusters = _cluster(
            df, self.k, self.max_epochs, self.distance_fn, self.init_c, self.verbose
        )
        return self

    def predict(self, df: pd.DataFrame):
        return _assign_to_clusters(
            df, _get_centers(df, self.clusters), self.distance_fn
        )


class Cluster:
    def __init__(self, centroid, rows):
        self.rows = rows
        self.centroid = centroid

    def __repr__(self):
        return f"Cluster(rows={self.rows}, centroid={self.centroid})"

    def __str__(self):
        return f"Cluster(rows={self.rows}, centroid={self.centroid})"

    def __eq__(self, other):
        return self.rows == other.rows and self.centroid == other.centroid

    def __hash__(self):
        return hash((self.centroid, self.rows))
