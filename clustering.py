import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import sys

DATA_FILE = 'data/locations.csv'
CLUSTERS = 16
FILE_NAME = "clustering_"

with open(DATA_FILE) as f:
    content = f.read().splitlines()
X = []
names = []
for row in content:
    data = row.split('","')
    v = [float(data[3]), float(data[2])]
    if v[0] != 0.0 and v[1] != 0:
        X.append(np.array(v))
        names.append(data[1])
X = np.array(X)

random_state = 170
y_pred = KMeans(n_clusters=CLUSTERS, random_state=random_state).fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("Lat / Lng")
plt.show()
plt.savefig(FILE_NAME + str(CLUSTERS) + '.png')
