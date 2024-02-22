import numpy as np
from sklearn.metrics import adjusted_rand_score

from utils import plot_confusion_matrix, show_image_mnist
from kmeans import KMeansClusterer

# Loading in the data
data = np.load("data/data.npy")
labels = np.load("data/labels.npy") #Note: given the labels, we will have 10 clusters.

# Show the first image
# show_image_mnist(data[0])