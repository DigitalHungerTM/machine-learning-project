import numpy as np
from sklearn.metrics import adjusted_rand_score

from utils import plot_confusion_matrix, show_image_mnist
from kmeans import KMeansClusterer

# Loading in the data
data = np.load("data/data.npy")
labels = np.load("data/labels.npy") #Note: given the labels, we will have 10 clusters.

# there are 10 labels (0-9)

# initialize 10 centroids

# Show the first image
# show_image_mnist(data[0])


clusterer = KMeansClusterer(n_clusters=10, n_feats=len(data[0]))
clusterer.initialize_clusters(data)
clusterer.assign(data)

