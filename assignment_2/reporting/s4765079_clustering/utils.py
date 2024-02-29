# utils.py
# Includes utility functions to help visualize and get a better understanding of the data.
# "moldy_frikandel" kernel Copyright (C) 2022 Lukas Edman - All Rights Reserved

from sklearn.manifold import TSNE
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import numpy as np

def show_image_mnist(image):
    """
    Converts an image to PIL format and displays the image.
        Args:
            image: np.array of shape [784], i.e. the flattened pixels in an MNIST image (28*28)
        Returns:
            None
    """
    from PIL import Image
    # normalize image with min-max if needed
    if image.max() > 1 or image.min() < 0:
        im = image - image.min()
        im = im / im.max()
    else:
        im = image

    im = Image.fromarray(np.uint8(np.reshape(im, (28, 28))*255))
    im.show()


def plot_tsne(data, *labels):
    """ 
        Runs TSNE on the first 1000 examples and plots the first 1000.
        Example usage: 
            plot_tsne(data, predictions1, labels)
            plot_tsne(data, predictions1, predictions2, labels)
            plot_tsne(data, predictions1, predictions2, ..., labels)

        Args:
            data:   np array of shape [N x F], 
                    N = number of data samples 
                    F = number of features, e.g. pixels in a flattened MNIST image (784)
            labels: np arrays, each of shape [N]. Providing multiple lists can be used 
                    for plotting multiple graphs, e.g. plotting the predicted labels vs true labels
        Returns:
            None
    """
    tsne = TSNE(n_components=2, init='pca', verbose=1)
    t_embs = tsne.fit_transform(data[:1000])
    fig, ax = plt.subplots(1, len(labels))
    for i, lab in enumerate(labels):
        ax[i].scatter(t_embs[:, 0], t_embs[:, 1], c=lab[:1000], cmap="tab10", marker=".")
    plt.show()

def plot_confusion_matrix(labels, predictions):
    """ 
        Creates and plots a confusion matrix.
        Args:
            labels:      np array of shape [N],
                         N = number of data samples 
            predictions: np array of shape [N].
        Returns:
            None
    """
    cmx = contingency_matrix(labels, predictions)
    disp = ConfusionMatrixDisplay(cmx)
    disp.plot()
    plt.show()


def get_kernel(kernel_name):
    """
    Returns a convolutional kernel given a name.
    Gaussian blurs can have any kernel size.
    Sobel filters can have any odd kernel size.
    The first value after "sobel_" is the angle of rotation in degrees. 
    Possible names:
        gaussian_blur_3x3
        gaussian_blur_4x4
        gaussian_blur_5x5
        ...etc.
        sobel_0_3x3
        sobel_0_5x5
        sobel_0_7x7
        sobel_15_3x3
        sobel_30_3x3
        sobel_45_3x3
        ...etc.
        moldy_frikandel (the one and only)
    Args:
        kernel_name: A string corresponding to one of the names listed
    Returns:
        kernel - a 2D np array 
    """
    if "gaussian_blur" in kernel_name:
        # This block of code generates any size gaussian kernel, for example if 
        # you give as input "gaussian_blur_10x10", it will give a 10x10 kernel.
        # You can see what the 3x3 and 5x5 kernels look like in the dict below
        # Example outputs:
        #     "gaussian_blur_3x3": np.array([[1,2,1],
        #                                    [2,4,2],
        #                                    [1,2,1]])/16,
        #     "gaussian_blur_5x5": np.array([[1,4,6,4,1],
        #                                    [4,16,24,16,4],
        #                                    [6,24,36,24,6],
        #                                    [4,16,24,16,4],
        #                                    [1,4,6,4,1]])/256,
        width = int(kernel_name.split("x")[-1])
        from scipy.linalg import pascal
        m = np.array(np.diag(pascal(width)[::-1]))[None]
        m2 = np.repeat(m, width, axis=0)
        kernel = m.T * m2
        return kernel / kernel.sum()
        
    elif "sobel" in kernel_name and "x" in kernel_name:
        # This code lets you use a sobel filter of any odd size, for any direction
        # giving the name "sobel_90_3x3" will give you a 3x3 horizontal edge filter,
        # rotated by 90 degrees, thus making it a vertical edge filter.
        width = int(kernel_name.split("x")[-1])
        alpha = int(kernel_name.split("_")[1])

        def create_sobel_kernel(width, alpha):
            assert width % 2 == 1, "width must be odd."
            kernel = np.zeros((width, width))
            for i in range(width//2+1):
                for j in range(width//2+1):
                    val = (i * np.cos(np.deg2rad(alpha)) + 
                           j * np.sin(np.deg2rad(alpha)))/max(1,(i*i + j*j))
                    kernel[width//2+i,width//2+j] = val
                    kernel[width//2+i,width//2-j] = val
                    kernel[width//2-i,width//2-j] = -val
                    kernel[width//2-i,width//2+j] = -val
            return kernel
        return create_sobel_kernel(width, alpha)

    elif "moldy_frikandel" in kernel_name:
        # "You thought I was just a joke in class..."
        return np.array([[-1,2,-2],
                        [3,-5,5],
                        [-7,7,-7]])/-5
    else:
        assert False, "No kernel matching that name"

def convolve_data(data, kernel):
    """
    Convolves all of the data using a provided kernel
    Args:
        data:   np array of shape [N x F], 
                N = number of data samples 
                F = number of features, e.g. pixels in a flattened MNIST image (784)
        kernel: np array of shape [K, K]
                K = the kernel's width/height
    Returns:
        data - the convolved data. Its shape will remain the same.
    """
    new_data = []
    for i in range(len(data)):
        convd = convolve2d(np.reshape(data[i], (28, 28)), kernel, 'same').flatten()
        new_data.append(convd)

    new_data = np.stack(new_data, axis=0)
    return new_data
