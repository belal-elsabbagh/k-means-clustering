import numpy as np
import pandas as pd

from src.distance_functions import euclidean


def _init_centroids(df, k) -> dict:
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
    centers = {}
    for c_num, rows in clusters.items():
        c = df[df.index.isin(rows)]
        centers[c_num] = [np.mean(c[f])for f in df]
    return centers


def cluster(df: pd.DataFrame, k, distance_fn=None):
    distance_fn = _init_params(distance_fn)
    centroids = _init_centroids(df, k)
    while True:
        clusters = assign_to_clusters(df, centroids, distance_fn)
        centers = get_centers(df, clusters)
        if centers == centroids:
            return clusters
        centroids = centers


def _init_params(distance_fn):
    if distance_fn is None:
        distance_fn = euclidean
    return distance_fn
