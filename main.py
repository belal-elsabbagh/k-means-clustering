import pandas as pd

from src.cluster import cluster

if __name__ == '__main__':
    df = pd.read_csv('data/example.csv')
    starting_centroids = {'c1': [16], 'c2': [22], 'c3': [40]}
    print(cluster(df, init_c=starting_centroids))
