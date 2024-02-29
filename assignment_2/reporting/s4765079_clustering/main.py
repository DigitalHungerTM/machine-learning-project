import numpy as np
from sklearn.metrics import adjusted_rand_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from utils import show_image_mnist
from kmeans import KMeansClusterer


def plot_confusion_matrix(true, predicted):
    """
    My own implementation of plot_confusion_matrix, as the provided one in utils.py doesn't work
    Shows a colored plot of the confusion matrix
    
    :param `true`: ArrayLike of shape(n_samples), correct labels
    :param `predicted`: ArrayLike of shape(n_samples), predicted labels
    """
    cm = confusion_matrix(true, predicted)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.show()


def main():
    try:
        # Loading in the data
        # also do some type conversion
        data = [tuple(datapoint) for datapoint in np.load("data/data.npy")]
        labels = [int(label) for label in np.load("data/labels.npy")]

        clusterer = KMeansClusterer(n_clusters=len(set(labels)))

        ### Code for printing ari of a single run ###

        predictions = clusterer.fit_predict(data, labels)
        ari = adjusted_rand_score(labels, predictions)
        print(f"single run Adjusted Rand Score:\n{ari:.3f}")
        
        
        ### Code for obtaining ao5 of ari ###

        # aris = []
        # for _ in range(5):
        #     predictions = clusterer.fit_predict(data, labels)
        #     ari = adjusted_rand_score(labels, predictions)
        #     print("ARI:", ari)
        #     aris.append(ari)
        # print("avg of 5:", sum(aris)/5)


        ### Code for plotting the confusion matrix ###

        # predictions = clusterer.fit_predict(data, labels)
        # plot_confusion_matrix(labels, predictions)


    except KeyboardInterrupt:
        print("exiting")
        # raise
        exit(0)


if __name__ == "__main__":
    main()
