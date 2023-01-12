import pandas as pd

from src.cluster import cluster

if __name__ == '__main__':
    df = pd.read_csv('data/example.csv')
    k = 3
    print(cluster(
        df, k,
        init_c={
            'c1': [16],
            'c2': [22],
            'c3': [40]
        }))
