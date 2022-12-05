import pandas as pd

from src.cluster import cluster

if __name__ == '__main__':
    df = pd.read_csv('data/example.csv')
    k = 2
    print(cluster(df, k))
