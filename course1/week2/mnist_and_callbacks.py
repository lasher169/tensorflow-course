import os
import tensorflow as tf
from tensorflow import keras
from kaggle.api.kaggle_api_extended import KaggleApi
import shutil
import matplotlib.pyplot as plt
import string
import numpy as np
import random

# grader-required-cell

# Load the data

tf.config.set_visible_devices([], "GPU")

# Initialize the Kaggle API
api = KaggleApi()
# Authenticate with your Kaggle credentials
api.authenticate()

current_file_path = __file__
current_file_name = os.path.splitext(os.path.basename(current_file_path))[0]

dst = "data/"+ current_file_name

# Specify the dataset you want to download (e.g., "username/dataset-name")
dataset_name = "vikramtiwari/mnist-numpy"

# Specify the directory where you want to save the dataset
download_dir = dst

def cleanup(download_dir):
    if os.path.exists(download_dir):
        shutil.rmtree(download_dir)
        print(download_dir + " deleted")

cleanup(dst)

# Create the download directory if it doesn't exist
if not os.path.exists(download_dir):
    os.makedirs(download_dir)

api.dataset_download_files(dataset_name, path=download_dir, unzip=True)

# Get current working directory
current_dir = os.getcwd()
#
# # Append data/mnist.npz to the previous path to get the full path
data_path = os.path.join(current_dir, "data/"+current_file_name+"/mnist.npz")
#
# # Discard test set
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data(path=data_path)
#
# # Normalize pixel values
x_train = x_train / 255.0
#
# # grader-required-cell
#
data_shape = x_train.shape
#
print(f"There are {data_shape[0]} examples with shape ({data_shape[1]}, {data_shape[2]})")

random_numbers = [random.randint(1, 6000) for _ in range(5)]

#printing out stuff
# for x in random_numbers:
#     plt.imshow(x_train[x], cmap='gray')
#     plt.title('Label: {}'.format(y_train[x]))
#     plt.axis('off')
#     plt.show()

#
#
# # grader-required-cell
#
# # GRADED CLASS: myCallback
# ### START CODE HERE
#
# # Remember to inherit from the correct class
class myCallback(tf.keras.callbacks.Callback):
    # Define the correct function signature for on_epoch_end
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') >= 0.99:
            print("\nReached 99% accuracy so cancelling training!")

            # Stop training once the above condition is met
            self.model.stop_training = True

# ### END CODE HERE
#
# # grader-required-cell
#
# # GRADED FUNCTION: train_mnist
def train_mnist(x_train, y_train):
    ### START CODE HERE

    # Instantiate the callback class
    callbacks = myCallback()

    # Define the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Fit the model for 10 epochs adding the callbacks
    # and save the training history
    history = model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])


    ### END CODE HERE

    return history

# grader-required-cell

history = train_mnist(x_train, y_train)

# Plot the chart for accuracy and loss on both training and validation
acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
loss = history.history['loss']
# val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
# plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
# plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training loss')
plt.legend()

plt.show()