import tensorflow as tf
import os
import requests
import shutil
import io
from tensorflow.keras.layers import TextVectorization


training_zip_url = "https://storage.googleapis.com/tensorflow-1-public/course3/sarcasm.json"


tf.config.set_visible_devices([], "GPU")

import zipfile

current_file_path = __file__
current_file_name = os.path.splitext(os.path.basename(current_file_path))[0]

dst = "data/"+ current_file_name

def fetch(zip_url, extracted_dst):
    # Send a GET request to the URL
    response = requests.get(zip_url)

    if os.path.exists(extracted_dst):
        shutil.rmtree(extracted_dst)
        print(extracted_dst + " deleted")

    if not os.path.exists(extracted_dst):
        os.makedirs(extracted_dst)

    # Check if request was successful (status code 200)
    filename = os.path.join(extracted_dst, zip_url.split("/")[-1])

    # Check if request was successful (status code 200)
    if response.status_code == 200:
        with open(filename, 'wb') as file:
            file.write(response.content)
            print(f"File downloaded to: {filename}")
    else:
        print("Failed to download the zip file.")

fetch(training_zip_url, dst)

import json

filename = os.path.join(dst, training_zip_url.split("/")[-1])

# Load the JSON file
with open(filename, 'r') as f:
    datastore = json.load(f)

# Non-sarcastic headline
print(datastore[0])

# Sarcastic headline
print(datastore[20000])

# Initialize lists
sentences = []
labels = []

# Append elements in the dictionaries into each list
for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])

# Number of examples to use for training
training_size = 20000

# Vocabulary size of the tokenizer
vocab_size = 10000

# Maximum length of the padded sequences
max_length = 32

# Output dimensions of the Embedding layer
embedding_dim = 16

# Split the sentences
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]

# Split the labels
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

import numpy as np

#NOTE START do not adapt again for test, just run vectorization as seen below

# Initialize the Tokenizer class
vectorization = TextVectorization(max_tokens=vocab_size, output_mode='int', output_sequence_length=max_length)

#this is required before the line below
vectorization.adapt(training_sentences)
# Generate the word index dictionary for the training sentences
vectorized_training_sentences = vectorization(training_sentences)

# Generate and pad the testing sequences
vectorized_testing_sentences = vectorization(testing_sentences)

#NOTE END

vocabulary = vectorization.get_vocabulary()

# Convert the labels lists into numpy arrays
training_labels = np.array(training_labels)
testing_labels = np.array(testing_labels)

import tensorflow as tf

# Initialize a GlobalAveragePooling1D (GAP1D) layer
gap1d_layer = tf.keras.layers.GlobalAveragePooling1D()

# Define sample array
sample_array = np.array([[[10,2],[1,3],[1,1]]])

# Print shape and contents of sample array
print(f'shape of sample_array = {sample_array.shape}')
print(f'sample array: {sample_array}')

# Pass the sample array to the GAP1D layer
output = gap1d_layer(sample_array)

# Print shape and contents of the GAP1D output array
print(f'output shape of gap1d_layer: {output.shape}')
print(f'output array of gap1d_layer: {output.numpy()}')

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Print the model summary
model.summary()

# Compile the model
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

num_epochs = 30

# Train the model
history = model.fit(
    vectorized_training_sentences,
    training_labels,
    epochs=num_epochs,
    validation_data=(vectorized_testing_sentences, testing_labels), verbose=2)

import matplotlib.pyplot as plt


# Plot utility
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


# Plot the accuracy and loss
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

# Get the index-word dictionary
reverse_word_index = {index: word for index, word in enumerate(vocabulary)}

# Get the embedding layer from the model (i.e. first layer)
embedding_layer = model.layers[0]

# Get the weights of the embedding layer
embedding_weights = embedding_layer.get_weights()[0]

# Print the shape. Expected is (vocab_size, embedding_dim)
print(embedding_weights.shape)

import io

# Open writeable files
out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')

# Initialize the loop. Start counting at `1` because `0` is just for the padding
for word_num in range(1, vocab_size):

  # Get the word associated at the current index
  word_name = reverse_word_index[word_num]

  # Get the embedding weights associated with the current index
  word_embedding = embedding_weights[word_num]

  # Write the word name
  out_m.write(word_name + "\n")

  # Write the word embedding
  out_v.write('\t'.join([str(x) for x in word_embedding]) + "\n")

# Close the files
out_v.close()
out_m.close()

