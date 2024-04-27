import os
from kaggle.api.kaggle_api_extended import KaggleApi
import shutil
import csv
import numpy as np
import matplotlib.pyplot as plt
import string
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img
import tensorflow as tf

# Initialize the Kaggle API
api = KaggleApi()
# Authenticate with your Kaggle credentials
api.authenticate()

dst = ""

# Specify the dataset you want to download (e.g., "username/dataset-name")
dataset_name = "datamunge/sign-language-mnist"

# Specify the directory where you want to save the dataset
download_dir = dst+"/assignment"

tf.config.set_visible_devices([], "GPU")

def cleanup(download_dir):
    if os.path.exists(download_dir):
        shutil.rmtree(download_dir)
        print(download_dir + " deleted")

cleanup(dst + "/assignment")

# Create the download directory if it doesn't exist
if not os.path.exists(download_dir):
    os.makedirs(download_dir)

api.dataset_download_files(dataset_name, path=download_dir, unzip=True)

TRAINING_FILE = dst+'/assignment/sign_mnist_train.csv'
VALIDATION_FILE = dst+'/assignment/sign_mnist_test.csv'

with open(TRAINING_FILE) as training_file:
  line = training_file.readline()
  print(f"First line (header) looks like this:\n{line}")
  line = training_file.readline()
  print(f"Each subsequent line (data points) look like this:\n{line}")

# grader-required-cell

# GRADED FUNCTION: parse_data_from_input
def parse_data_from_input(filename):
  """
  Parses the images and labels from a CSV file

  Args:
    filename (string): path to the CSV file

  Returns:
    images, labels: tuple of numpy arrays containing the images and labels
  """
  with open(filename) as file:
    ### START CODE HERE

    # Use csv.reader, passing in the appropriate delimiter
    # Remember that csv.reader can be iterated and returns one line in each iteration
    # data = np.loadtxt(TRAINING_FILE, ',', skiprows=1)

    # labels = data[:, 1]
    # images = data[:, 1:]
    csv_reader = csv.reader(file, delimiter=',')

    next(csv_reader)

    labels = []
    images = []

    for row in csv_reader:
        # Extract label from the first column
        labels.append(float(row[0]))  # Convert to appropriate data type if needed

        # Extract image data from the remaining columns
        image_row = [float(val) for val in row[1:]]  # Convert to appropriate data type if needed
        images.append(np.array_split(image_row, 28)) #convert from 784 to 28x28



    # print("Labels:", labels)
    # print("Image Data:", images)
    # Convert lists to NumPy arrays
    labels = np.array(labels)
    images = np.array(images)

    ### END CODE HERE

    return images, labels

# grader-required-cell

# Test your function
training_images, training_labels = parse_data_from_input(TRAINING_FILE)
validation_images, validation_labels = parse_data_from_input(VALIDATION_FILE)
#
print(f"Training images has shape: {training_images.shape} and dtype: {training_images.dtype}")
print(f"Training labels has shape: {training_labels.shape} and dtype: {training_labels.dtype}")
print(f"Validation images has shape: {validation_images.shape} and dtype: {validation_images.dtype}")
print(f"Validation labels has shape: {validation_labels.shape} and dtype: {validation_labels.dtype}")

# Plot a sample of 10 images from the training set
def plot_categories(training_images, training_labels):
  fig, axes = plt.subplots(1, 10, figsize=(16, 15))
  axes = axes.flatten()
  letters = list(string.ascii_lowercase)

  for k in range(10):
    img = training_images[k]
    img = np.expand_dims(img, axis=-1)
    img = array_to_img(img)
    ax = axes[k]
    ax.imshow(img, cmap="Greys_r")
    ax.set_title(f"{letters[int(training_labels[k])]}")
    ax.set_axis_off()

  plt.tight_layout()
  plt.show()

# plot_categories(training_images, training_labels)

# grader-required-cell

# GRADED FUNCTION: train_val_generators
def train_val_generators(training_images, training_labels, validation_images, validation_labels):
  """
  Creates the training and validation data generators

  Args:
    training_images (array): parsed images from the train CSV file
    training_labels (array): parsed labels from the train CSV file
    validation_images (array): parsed images from the test CSV file
    validation_labels (array): parsed labels from the test CSV file

  Returns:
    train_generator, validation_generator - tuple containing the generators
  """
  ### START CODE HERE

  # In this section you will have to add another dimension to the data
  # So, for example, if your array is (10000, 28, 28)
  # You will need to make it (10000, 28, 28, 1)
  # Hint: np.expand_dims
  training_images = np.expand_dims(training_images, axis=-1)
  validation_images = np.expand_dims(validation_images, axis=-1)

  # Instantiate the ImageDataGenerator class
  # Don't forget to normalize pixel values
  # and set arguments to augment the images (if desired)
  train_datagen = ImageDataGenerator(
      rescale = 1./255.,
	  rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')


  # Pass in the appropriate arguments to the flow method
  train_generator = train_datagen.flow(x=training_images,
                                       y=training_labels,
                                       batch_size=32)


  # Instantiate the ImageDataGenerator class (don't forget to set the rescale argument)
  # Remember that validation data should not be augmented
  validation_datagen = ImageDataGenerator(rescale = 1./255.)

  # Pass in the appropriate arguments to the flow method
  validation_generator = validation_datagen.flow(x=validation_images,
                                                 y=validation_labels,
                                                 batch_size=32)

  ### END CODE HERE

  return train_generator, validation_generator

# grader-required-cell

# Test your generators
train_generator, validation_generator = train_val_generators(training_images, training_labels, validation_images, validation_labels)

print(f"Images of training generator have shape: {train_generator.x.shape}")
print(f"Labels of training generator have shape: {train_generator.y.shape}")
print(f"Images of validation generator have shape: {validation_generator.x.shape}")
print(f"Labels of validation generator have shape: {validation_generator.y.shape}")

# grader-required-cell

def create_model():

  ### START CODE HERE

  # Define the model
  # Use no more than 2 Conv2D and 2 MaxPooling2D
  # model = tf.keras.models.Sequential([
  #   # This is the first convolution
  #   tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
  #   tf.keras.layers.MaxPooling2D(2, 2),
  #   # The second convolution
  #   tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
  #   tf.keras.layers.MaxPooling2D(2,2),
  #
  #   # Flatten the results to feed into a DNN
  #   tf.keras.layers.Flatten(),
  #   tf.keras.layers.Dropout(0.5),
  #
  #   # 128 neuron hidden layer
  #   tf.keras.layers.Dense(128, activation='relu'),
  #
  #   tf.keras.layers.Dense(25, activation='softmax')
  # ])
  # Define the model
  model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The second convolution
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(25, activation='softmax')
  ])

  from tensorflow.keras.optimizers.legacy import Adam

  model.compile(optimizer='rmsprop',
                loss = 'sparse_categorical_crossentropy',
                metrics=['accuracy'])

  ### END CODE HERE

  return model

# Save your model
model = create_model()

# Train your model
history = model.fit(train_generator,
                    epochs=15,
                    validation_data=validation_generator,
                    verbose=1)

# Plot the chart for accuracy and loss on both training and validation
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()