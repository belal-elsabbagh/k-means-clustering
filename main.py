import pandas as pd

from src.cluster import KMeans

if __name__ == '__main__':
    df = pd.read_csv('data/example.csv')
    starting_centroids = {'c1': [16], 'c2': [22], 'c3': [40]}
    model = KMeans(k=3, init_c=None, verbose=True)
    model.fit(df)
    print(f'Clusters: {model.clusters}')
