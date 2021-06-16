# Permission (to download dataset from internet)
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import tensorflow as tf
import numpy as np
from tensorflow import keras

# Printing Stuff
import matplotlib.pyplot as plt

# Load a pre-defined dataset (70k of 28*28)
fashion_mnist = keras.datasets.fashion_mnist

# here pulling the data from dataset.
(train_image, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# show data
print(train_labels[0])

for img, label in train_image, train_labels:
    print(label)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.show()
