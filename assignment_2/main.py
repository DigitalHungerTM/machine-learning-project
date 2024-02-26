import numpy as np
from sklearn.metrics import adjusted_rand_score
from pprint import pprint

from utils import plot_confusion_matrix, show_image_mnist
from kmeans import KMeansClusterer

# Loading in the data
# also do some type conversion
data = [tuple(datapoint) for datapoint in np.load("data/data.npy")]
labels = [int(label) for label in np.load("data/labels.npy")]

clusterer = KMeansClusterer(n_clusters=len(set(labels)), n_feats=len(data[0]))

aris = []
for i in range(5):
    predictions = clusterer.fit_predict(data, labels)
    ari = adjusted_rand_score(labels, predictions)
    print("ARI:", ari)
    aris.append(ari)

print("avg of 5:", sum(aris)/5)
