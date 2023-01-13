import numpy as np
import pandas as pd

from src.distance_functions import euclidean

VALUE_ERROR_MSG = 'Either k or the initial cluster (init_c) must be provided'


def _init_c(df, k) -> dict:
    return {i+1: row[1].tolist() for i, row in zip(range(k), df.sample(n=k).iterrows())}


def assign_to_clusters(df, centroids, distance_fn):
    rows = {}
    clusters = {}
    for index, row in df.iterrows():
        d = {ind: distance_fn(row.tolist(), c) for ind, c in centroids.items()}
        rows[index] = min(d, key=d.get)
    return group_rows_by_cluster(clusters, rows)


def group_rows_by_cluster(clusters, rows):
    for i, v in rows.items():
        clusters[v] = [i] if v not in clusters.keys() else clusters[v] + [i]
    return clusters


def get_centers(df, clusters) -> dict:
    if clusters is None:
        return {}
    centers = {}
    for c_num, rows in clusters.items():
        c = df[df.index.isin(rows)]
        centers[c_num] = [np.mean(c[f])for f in df]
    return centers


def cluster(df: pd.DataFrame, k=None, distance_fn=None, init_c=None):
    distance_fn, centroids = _init_params(df, k, distance_fn, init_c)
    centers = None
    clusters = None
    while centers != centroids:
        centroids = centers if centers is not None else centroids
        print(centroids)
        clusters = assign_to_clusters(df, centroids, distance_fn)
        centers = get_centers(df, clusters)
    return clusters


def _init_params(df, k, distance_fn, init_c):
    if init_c is None and k is None:
        raise ValueError(VALUE_ERROR_MSG)
    return (
        euclidean if distance_fn is None else distance_fn,
        _init_c(df, k) if init_c is None else init_c
    )
