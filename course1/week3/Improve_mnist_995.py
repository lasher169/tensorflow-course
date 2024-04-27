# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# import time
#
# # grader-required-cell
#
# # Load the data
#
# # Get current working directory
# current_dir = os.getcwd()
#
# tf.config.set_visible_devices([], 'GPU')
#
# # Append data/mnist.npz to the previous path to get the full path
# data_path = os.path.join(current_dir, "data/mnist.npz")
#
# # Get only training set
# (training_images, training_labels), _ = tf.keras.datasets.mnist.load_data(path=data_path)


import os
import tensorflow as tf
from tensorflow import keras
from kaggle.api.kaggle_api_extended import KaggleApi
import shutil
import matplotlib.pyplot as plt
import string
import numpy as np
import random
import time

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
(training_images, training_labels), _ = tf.keras.datasets.mnist.load_data(path=data_path)

# grader-required-cell

# GRADED FUNCTION: reshape_and_normalize

def reshape_and_normalize(images):
    ### START CODE HERE

    # Reshape the images to add an extra dimension
    images = np.expand_dims(images, axis=-1)

    # Normalize pixel values
    images = images / 255

    ### END CODE HERE

    return images

# grader-required-cell

# Reload the images in case you run this cell multiple times
(training_images, _), _ = tf.keras.datasets.mnist.load_data(path=data_path)

# Apply your function
training_images = reshape_and_normalize(training_images)

print(f"Maximum pixel value after normalization: {np.max(training_images)}\n")
print(f"Shape of training set after reshaping: {training_images.shape}\n")
print(f"Shape of one image after reshaping: {training_images[0].shape}")


# grader-required-cell

# GRADED CLASS: myCallback
### START CODE HERE

# Remember to inherit from the correct class
class myCallback(tf.keras.callbacks.Callback):
    # Define the method that checks the accuracy at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > .995):
            # Stop if threshold is met
            print("\nAccuracy is greater than 99.5 so cancelling training!")
            self.model.stop_training = True


callbacks = myCallback()

### END CODE HERE

# grader-required-cell

# GRADED FUNCTION: convolutional_model
def convolutional_model():
    ### START CODE HERE



    # Define the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),


        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    ### END CODE HERE

    # Compile the model
    model.compile(optimizer='RMSprop',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# grader-required-cell

# Save your untrained model


model = convolutional_model()


# Get number of weights
model_params = model.count_params()

# Unit test to limit the size of the model
assert model_params < 1000000, (
    f'Your model has {model_params:,} params. For successful grading, please keep it ' 
    f'under 1,000,000 by reducing the number of units in your Conv2D and/or Dense layers.'
)

# Instantiate the callback class
callbacks = myCallback()

# Train your model (this can take up to 5 minutes)

# Record end time
start_time = time.time()

history = model.fit(training_images, training_labels, epochs=20, callbacks=[callbacks])

end_time = time.time()
# Calculate the difference
time_difference = end_time - start_time

# Print the time difference
print("Time difference:", time_difference, "seconds")

# grader-required-cell

print(f"Your model was trained for {len(history.epoch)} epochs")

if not "accuracy" in history.model.metrics_names:
    print("Use 'accuracy' as metric when compiling your model.")
else:
    print("The metric was correctly defined.")