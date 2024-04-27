import io
import os
import random
import shutil
from shutil import copyfile
import zipfile

import matplotlib.pyplot as plt
import requests
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

training_zip_url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
training_extracted_dst = "../week2"

def fetch_and_extract_zip(zip_url, extracted_dst):
    # Send a GET request to the URL
    response = requests.get(zip_url)

    # Check if request was successful (status code 200)
    if response.status_code == 200:

        if os.path.exists(extracted_dst):
            shutil.rmtree(extracted_dst)
            print(extracted_dst + " deleted")
        # Unzip the dataset
        with zipfile.ZipFile(io.BytesIO(response.content), 'r') as zip_ref:
            # Extract all contents to a directory
            zip_ref.extractall(extracted_dst)
        print("Zip file extracted successfully.")
    else:
        print("Failed to download the zip file.")


fetch_and_extract_zip(training_zip_url, training_extracted_dst+"/assignment")

source_path = '/PetImages'

source_path_dogs = os.path.join(training_extracted_dst+"/assignment"+source_path, 'Dog')
source_path_cats = os.path.join(training_extracted_dst+"/assignment"+source_path, 'Cat')

# Deletes all non-image files (there are two .db files bundled into the dataset)
#!find /tmp/PetImages/ -type f ! -name "*.jpg" -exec rm {} +

if os.path.exists(training_extracted_dst+"/assignment"+source_path) and os.path.isdir(training_extracted_dst+"/assignment"+source_path):
    for root, dirs, files in os.walk(training_extracted_dst+"/assignment"+source_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            # Check if the file does not end with ".jpg"
            if not filename.lower().endswith('.jpg'):
                # Delete the file
                os.remove(file_path)
                print(f"Deleted: {file_path}")

# os.listdir returns a list containing all files under the given path
print(f"There are {len(os.listdir(source_path_dogs))} images of dogs.")
print(f"There are {len(os.listdir(source_path_cats))} images of cats.")

# grader-required-cell

# Define root directory
root_dir = training_extracted_dst+"/assignment"+'/cats-v-dogs'

# Empty directory to prevent FileExistsError is the function is run several times
if os.path.exists(root_dir):
  shutil.rmtree(root_dir)

# GRADED FUNCTION: create_train_val_dirs
def create_train_val_dirs(root_path):
  """
  Creates directories for the train and test sets

  Args:
    root_path (string) - the base directory path to create subdirectories from

  Returns:
    None
  """

  ### START CODE HERE

  # HINT:
  # Use os.makedirs to create your directories with intermediate subdirectories
  # Don't hardcode the paths. Use os.path.join to append the new directories to the root_path parameter

  training = "training"
  validation = "validation"
  cat = "cats"
  dog = "dogs"

  training_dir = os.path.join(root_path, training)
  dog_training = os.path.join(training_dir, dog)
  cat_training = os.path.join(training_dir, cat)

  validation_dir = os.path.join(root_path, validation)
  dog_validation = os.path.join(validation_dir, dog)
  cat_validation = os.path.join(validation_dir, cat)

  os.makedirs(dog_training)
  os.makedirs(cat_training)
  os.makedirs(dog_validation)
  os.makedirs(cat_validation)


  ### END CODE HERE


try:
  create_train_val_dirs(root_path=root_dir)
except FileExistsError:
  print("You should not be seeing this since the upper directory is removed beforehand")

# grader-required-cell

# Test your create_train_val_dirs function

for rootdir, dirs, files in os.walk(root_dir):
    for subdir in dirs:
        print(os.path.join(rootdir, subdir))

# grader-required-cell

# GRADED FUNCTION: split_data
def split_data(SOURCE_DIR, TRAINING_DIR, VALIDATION_DIR, SPLIT_SIZE):
  """
  Splits the data into train and test sets

  Args:
    SOURCE_DIR (string): directory path containing the images
    TRAINING_DIR (string): directory path to be used for training
    VALIDATION_DIR (string): directory path to be used for validation
    SPLIT_SIZE (float): proportion of the dataset to be used for training

  Returns:
    None
  """
  ### START CODE HERE
  files = os.listdir(SOURCE_DIR)
  random.sample(files, len(files))

  split_index = int(len(files) * SPLIT_SIZE)

  train_files = files[:split_index]
  valid_files = files[split_index:]

  for file in train_files:
      file_size = os.path.getsize(os.path.join(SOURCE_DIR, file))
      if file_size > 0:
          copyfile(os.path.join(SOURCE_DIR, file), os.path.join(TRAINING_DIR, file))
      else:
          print(file, "is zero length, so ignoring.")

  for file in valid_files:
      file_size = os.path.getsize(os.path.join(SOURCE_DIR, file))
      if file_size > 0:
          copyfile(os.path.join(SOURCE_DIR, file), os.path.join(VALIDATION_DIR, file))
      else:
          print(file, "is zero length, so ignoring.")

  # new_train_files = os.listdir(TRAINING_DIR)
  # for file in new_train_files:
  #     print(file)


  ### END CODE HERE

# grader-required-cell

# Test your split_data function

# Define paths
CAT_SOURCE_DIR = training_extracted_dst+"/assignment/PetImages/Cat/"
DOG_SOURCE_DIR = training_extracted_dst+"/assignment/PetImages/Dog/"

TRAINING_DIR = training_extracted_dst+"/assignment/cats-v-dogs/training/"
VALIDATION_DIR = training_extracted_dst+"/assignment/cats-v-dogs/validation/"

TRAINING_CATS_DIR = os.path.join(TRAINING_DIR, "cats/")
VALIDATION_CATS_DIR = os.path.join(VALIDATION_DIR, "cats/")

TRAINING_DOGS_DIR = os.path.join(TRAINING_DIR, "dogs/")
VALIDATION_DOGS_DIR = os.path.join(VALIDATION_DIR, "dogs/")

# Empty directories in case you run this cell multiple times
if len(os.listdir(TRAINING_CATS_DIR)) > 0:
  for file in os.scandir(TRAINING_CATS_DIR):
      os.remove(file.path)
if len(os.listdir(TRAINING_DOGS_DIR)) > 0:
  for file in os.scandir(TRAINING_DOGS_DIR):
      os.remove(file.path)
if len(os.listdir(VALIDATION_CATS_DIR)) > 0:
  for file in os.scandir(VALIDATION_CATS_DIR):
      os.remove(file.path)
if len(os.listdir(VALIDATION_DOGS_DIR)) > 0:
  for file in os.scandir(VALIDATION_DOGS_DIR):
      os.remove(file.path)

# Define proportion of images used for training
split_size = .9

# Run the function
# NOTE: Messages about zero length images should be printed out
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, VALIDATION_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, VALIDATION_DOGS_DIR, split_size)

# Your function should perform copies rather than moving images so original directories should contain unchanged images
print(f"\n\nOriginal cat's directory has {len(os.listdir(CAT_SOURCE_DIR))} images")
print(f"Original dog's directory has {len(os.listdir(DOG_SOURCE_DIR))} images\n")

# Training and validation splits. Check that the number of images matches the expected output.
print(f"There are {len(os.listdir(TRAINING_CATS_DIR))} images of cats for training")
print(f"There are {len(os.listdir(TRAINING_DOGS_DIR))} images of dogs for training")
print(f"There are {len(os.listdir(VALIDATION_CATS_DIR))} images of cats for validation")
print(f"There are {len(os.listdir(VALIDATION_DOGS_DIR))} images of dogs for validation")

# grader-required-cell

# GRADED FUNCTION: train_val_generators
def train_val_generators(TRAINING_DIR, VALIDATION_DIR):
  """
  Creates the training and validation data generators

  Args:
    TRAINING_DIR (string): directory path containing the training images
    VALIDATION_DIR (string): directory path containing the testing/validation images

  Returns:
    train_generator, validation_generator - tuple containing the generators
  """
  ### START CODE HERE

  # Instantiate the ImageDataGenerator class (don't forget to set the arguments to augment the images)
  train_datagen = ImageDataGenerator(
      rescale=1. / 255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

  # Pass in the appropriate arguments to the flow_from_directory method
  train_generator = train_datagen.flow_from_directory(directory=TRAINING_DIR,
                                                      batch_size=128,
                                                      class_mode='binary',
                                                      target_size=(150, 150))

  # Instantiate the ImageDataGenerator class (don't forget to set the rescale argument)
  validation_datagen = ImageDataGenerator(rescale=1. / 255)


  # Pass in the appropriate arguments to the flow_from_directory method
  validation_generator = validation_datagen.flow_from_directory(directory=VALIDATION_DIR,
                                                                batch_size=32,
                                                                class_mode='binary',
                                                                target_size=(150, 150))
  ### END CODE HERE
  return train_generator, validation_generator

# grader-required-cell

# Test your generators
train_generator, validation_generator = train_val_generators(TRAINING_DIR, VALIDATION_DIR)

# grader-required-cell

# GRADED FUNCTION: create_model
def create_model():
  # DEFINE A KERAS MODEL TO CLASSIFY CATS V DOGS
  # USE AT LEAST 3 CONVOLUTION LAYERS

  ### START CODE HERE

  model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
      tf.keras.layers.MaxPooling2D(2, 2),
      tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
      tf.keras.layers.MaxPooling2D(2, 2),
      tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
      tf.keras.layers.MaxPooling2D(2, 2),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(512, activation='relu'),
      tf.keras.layers.Dense(1, activation='sigmoid')
  ])

  from tensorflow.keras.optimizers.legacy import RMSprop

  model.compile(loss='binary_crossentropy',
                optimizer=RMSprop(learning_rate=1e-3),
                metrics=['accuracy'])

  ### END CODE HERE

  return model

# Get the untrained model
model = create_model()

# Train the model
# Note that this may take some time.
history = model.fit(train_generator,
                    epochs=15,
                    # steps_per_epoch=100,  # 2000 images = batch_size * steps
                    verbose=1,
                    validation_data=validation_generator)
                    # validation_steps=50)  # 1000 images = batch_size * steps)

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.show()
print("")

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")
plt.show()