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

# # show data
# for index in range(0, len(train_image)):
#     print(train_labels[index])
#     plt.imshow(train_image[index], cmap='gray', vmin=0, vmax=255)
#     plt.show()


# define out neural net structure.
model = keras.Sequentail([

    # input is 28x28 image ("Flatten" flattens the 28x28 into a single input 28*28  = 784 input layer
    keras.layers.Flatten(input_shape=(28, 28)),

    

])
