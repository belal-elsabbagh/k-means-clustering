from matplotlib import pyplot as plt
import pandas as pd
from src.cluster import KMeans


df = pd.read_csv("data/assignment-example.csv", index_col="id")
sse = []
for k in range(1, 9):
    model = KMeans(k=k, verbose=True)
    model.fit(df)
    res = [model.predict(i) for i in df.values]
    s = [model.distance_fn(i, model.centers[r]) for i, r in zip(df.values, res)]
    print(f"Sum of squared errors: {sum(s)}")
    sse.append(sum(s))
plt.plot(range(1, 9), sse)
plt.xlabel("k")
plt.ylabel("SSE")
plt.show()
print(f"Centers: {model.centers}")
