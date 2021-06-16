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
model = keras.Sequential([

    # input is 28x28 image ("Flatten" flattens the 28x28 into a single input 28*28  = 784 input layer
    keras.layers.Flatten(input_shape=(28, 28)),

    # hidden layer is 128 deep. relu returns the value, or 0 (works good enough. much faster)
    keras.layers.Dense(units=128, activation=tf.nn.relu),

    # output is 0-10 (depending on what piece of clothing it is). return maximum.
    keras.layers.Dense(units=10, activation=tf.nn.softmax)
])

# compile our model.
model.compile(optimizer=tf.optimizers.Adam(), loss="sparse_categorical_crossentropy")

# train our model using training data set.
model.fit(train_image, train_labels, epochs=5)

# Test our model using our testing data.
test_loss = model.evaluate(test_images, test_labels)

# Make Prediction
predictions = model.predict(test_images)

# Print out our prediction.
print("Predicted By The neural NetWork: ", list(predictions[2]).index(max(predictions[2])))
plt.imshow(test_images[2], cmap='gray', vmin=0, vmax=255)
plt.show()
print("Right Label From data Set: ", test_labels[2])
