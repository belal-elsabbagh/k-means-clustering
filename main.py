import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.cluster import KMeans, hierarchical_clustering

if __name__ == "__main__":
    df = pd.read_csv("data/assignment-example.csv", index_col="id")
    print(hierarchical_clustering(df))
