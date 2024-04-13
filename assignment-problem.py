from matplotlib import pyplot as plt
import pandas as pd
from src.cluster import KMeans


df = pd.read_csv("data/assignment-example.csv", index_col="id")
starting_centroids = {1: [2, 2], 2: [1, 1]}
model = KMeans(init_c=starting_centroids, verbose=True)
model.fit(df)
res = [model.predict(i) for i in df.values]
plt.scatter(df["x"], df["y"], c=res, cmap="viridis")
plt.scatter(model.center_dim(0), model.center_dim(1), c="red", marker="x")
for i in df.index:
    plt.annotate(i, (df["x"][i].item() + 0.02, df["y"][i].item() + 0.02))
plt.show()
