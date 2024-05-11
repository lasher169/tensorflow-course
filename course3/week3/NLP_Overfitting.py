# grader-required-cell

import csv
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import zipfile
import io
import requests
import os
import shutil

# grader-required-cell

EMBEDDING_DIM = 100
MAXLEN = 16
TRUNCATING = 'post'
PADDING = 'post'
OOV_TOKEN = "<OOV>"
MAX_EXAMPLES = 160000
TRAINING_SPLIT = 0.9

# grader-required-cell

zip_url = "https://drive.usercontent.google.com/download?id=1p_4fA2RAwrkdasgMoeZBGDsgWLNG3PK7&export=download&authuser=1&confirm=t"

SENTIMENT_CSV = "./data/training_cleaned.csv"

tf.config.set_visible_devices([], 'GPU')

dst = "data/"

def cleanup(extracted_dst):
    if os.path.exists(extracted_dst):
        shutil.rmtree(extracted_dst)
        print(extracted_dst + " deleted")

cleanup(dst)


# Define a function to download and extract the zip file
def fetch_and_extract_zip(zip_url, dst, filename):
    # Send a GET request to the URL
    response = requests.get(zip_url)

    # Check if request was successful (status code 200)
    if response.status_code == 200:
        # Unzip the dataset
        file_content = response.content
        # Create the folder if it doesn't exist
        os.makedirs(dst, exist_ok=True)

        # Save the file to a local file
        # Save the file to the specified folder
        with open(os.path.join(dst, filename), 'wb') as f:
            f.write(file_content)
        print("File downloaded successfully.")
    else:
        print("Failed to download the file.")

fetch_and_extract_zip(zip_url, dst, 'training_cleaned.csv')

with open(SENTIMENT_CSV, 'r') as csvfile:
    print(f"First data point looks like this:\n\n{csvfile.readline()}")
    print(f"Second data point looks like this:\n\n{csvfile.readline()}")


#Parsing the raw data Now you need to read the data from the csv file.
# To do so, complete the parse_data_from_file function.
# A couple of things to note:You should NOT omit the first line as the file does not contain headers.
# There is no need to save the data points as numpy arrays, regular lists is fine.
# To read from csv files use csv.reader by passing the appropriate arguments.
# csv.reader returns an iterable that returns each row in every iteration.
# So the label can be accessed via row[0] and the text via row[5].
# The labels are originally encoded as strings ('0' representing negative and '4' representing positive).
# You need to change this so that the labels are integers and 0 is used for representing negative, while 1 should represent positive.

# grader-required-cell

# GRADED FUNCTION: parse_data_from_file
def parse_data_from_file(filename):
    """
    Extracts sentences and labels from a CSV file

    Args:
        filename (string): path to the CSV file

    Returns:
        sentences, labels (list of string, list of string): tuple containing lists of sentences and labels
    """

    sentences = []
    labels = []

    with open(filename, 'r') as csvfile:
        ### START CODE HERE
        reader = csv.reader(csvfile, delimiter=",")
        print()
        for row in reader:
            if row[0] == "0":
                labels.append(0)
            else:
                labels.append(1)
            sentences.append(row[5])
        ### END CODE HERE

    return sentences, labels

parse_data_from_file(SENTIMENT_CSV)

# grader-required-cell

# Test your function
sentences, labels = parse_data_from_file(SENTIMENT_CSV)

print(f"dataset contains {len(sentences)} examples\n")

print(f"Text of second example should look like this:\n{sentences[1]}\n")
print(f"Text of fourth example should look like this:\n{sentences[3]}")

print(f"\nLabels of last 5 examples should look like this:\n{labels[-5:]}")

# grader-required-cell

# Bundle the two lists into a single one
sentences_and_labels = list(zip(sentences, labels))

# Perform random sampling
random.seed(42)
sentences_and_labels = random.sample(sentences_and_labels, MAX_EXAMPLES)

# Unpack back into separate lists
sentences, labels = zip(*sentences_and_labels)

print(f"There are {len(sentences)} sentences and {len(labels)} labels after random sampling\n")


# grader-required-cell

# GRADED FUNCTION: train_val_split
def train_val_split(sentences, labels, training_split):
    """
    Splits the dataset into training and validation sets

    Args:
        sentences (list of string): lower-cased sentences without stopwords
        labels (list of string): list of labels
        training split (float): proportion of the dataset to convert to include in the train set

    Returns:
        train_sentences, validation_sentences, train_labels, validation_labels - lists containing the data splits
    """
    ### START CODE HERE

    # Compute the number of sentences that will be used for training (should be an integer)
    train_size = len(sentences)

    # Split the sentences and labels into train/validation splits

    split_size = int(train_size * training_split)

    train_sentences = sentences[:split_size]
    train_labels = labels[:split_size]

    validation_sentences = sentences[split_size:]
    validation_labels = labels[split_size:]

    ### END CODE HERE

    return train_sentences, validation_sentences, train_labels, validation_labels

# grader-required-cell

# Test your function
train_sentences, val_sentences, train_labels, val_labels = train_val_split(sentences, labels, TRAINING_SPLIT)

print(f"There are {len(train_sentences)} sentences for training.\n")
print(f"There are {len(train_labels)} labels for training.\n")
print(f"There are {len(val_sentences)} sentences for validation.\n")
print(f"There are {len(val_labels)} labels for validation.")


# grader-required-cell

# GRADED FUNCTION: fit_tokenizer
def fit_tokenizer(train_sentences, oov_token):
    """
    Instantiates the Tokenizer class on the training sentences

    Args:
        train_sentences (list of string): lower-cased sentences without stopwords to be used for training
        oov_token (string) - symbol for the out-of-vocabulary token

    Returns:
        tokenizer (object): an instance of the Tokenizer class containing the word-index dictionary
    """
    ### START CODE HERE

    # Instantiate the Tokenizer class, passing in the correct values for oov_token
    tokenizer = Tokenizer(oov_token=oov_token)

    # Fit the tokenizer to the training sentences
    # Tokenize the input sentences
    tokenizer.fit_on_texts(train_sentences)

    ### END CODE HERE

    return tokenizer

# grader-required-cell

# Test your function
tokenizer = fit_tokenizer(train_sentences, OOV_TOKEN)

word_index = tokenizer.word_index
VOCAB_SIZE = len(word_index)

print(f"Vocabulary contains {VOCAB_SIZE} words\n")
print("<OOV> token included in vocabulary" if "<OOV>" in word_index else "<OOV> token NOT included in vocabulary")
print(f"\nindex of word 'i' should be {word_index['i']}")


# GRADED FUNCTION: seq_pad_and_trunc
def seq_pad_and_trunc(sentences, tokenizer, padding, truncating, maxlen):
    """
    Generates an array of token sequences and pads them to the same length

    Args:
        sentences (list of string): list of sentences to tokenize and pad
        tokenizer (object): Tokenizer instance containing the word-index dictionary
        padding (string): type of padding to use
        truncating (string): type of truncating to use
        maxlen (int): maximum length of the token sequence

    Returns:
        pad_trunc_sequences (array of int): tokenized sentences padded to the same length
    """
    ### START CODE HERE

    # Convert sentences to sequences
    sequences = tokenizer.texts_to_sequences(sentences)

    # Pad the sequences using the correct padding, truncating and maxlen
    pad_trunc_sequences = pad_sequences(sequences, maxlen=maxlen, truncating=truncating, padding=padding)

    ### END CODE HERE

    return pad_trunc_sequences

# grader-required-cell

# Test your function
train_pad_trunc_seq = seq_pad_and_trunc(train_sentences, tokenizer, PADDING, TRUNCATING, MAXLEN)
val_pad_trunc_seq = seq_pad_and_trunc(val_sentences, tokenizer, PADDING, TRUNCATING, MAXLEN)

print(f"Padded and truncated training sequences have shape: {train_pad_trunc_seq.shape}\n")
print(f"Padded and truncated validation sequences have shape: {val_pad_trunc_seq.shape}")

# grader-required-cell

train_labels = np.array(train_labels)
val_labels = np.array(val_labels)

# grader-required-cell

# Define path to file containing the embeddings
GLOVE_URL = 'https://drive.usercontent.google.com/download?id=1wf3VDxmOZktPzXwB3sG0YdVsk1eSPwXJ&export=download&authuser=1&confirm=t'
GLOVE_FILE = './data/glove.6B.100d.txt'

fetch_and_extract_zip(GLOVE_URL, dst, 'glove.6B.100d.txt')

# Initialize an empty embeddings index dictionary
GLOVE_EMBEDDINGS = {}

# Read file and fill GLOVE_EMBEDDINGS with its contents
with open(GLOVE_FILE) as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        GLOVE_EMBEDDINGS[word] = coefs

# grader-required-cell

test_word = 'dog'

test_vector = GLOVE_EMBEDDINGS[test_word]

print(f"Vector representation of word {test_word} looks like this:\n\n{test_vector}")

# grader-required-cell

print(f"Each word vector has shape: {test_vector.shape}")

# grader-required-cell

# Initialize an empty numpy array with the appropriate size
EMBEDDINGS_MATRIX = np.zeros((VOCAB_SIZE+1, EMBEDDING_DIM))

# Iterate all of the words in the vocabulary and if the vector representation for
# each word exists within GloVe's representations, save it in the EMBEDDINGS_MATRIX array
for word, i in word_index.items():
    embedding_vector = GLOVE_EMBEDDINGS.get(word)
    if embedding_vector is not None:
        EMBEDDINGS_MATRIX[i] = embedding_vector


# grader-required-cell

# GRADED FUNCTION: create_model
def create_model(vocab_size, embedding_dim, maxlen, embeddings_matrix):
    """
    Creates a binary sentiment classifier model

    Args:
        vocab_size (int): size of the vocabulary for the Embedding layer input
        embedding_dim (int): dimensionality of the Embedding layer output
        maxlen (int): length of the input sequences
        embeddings_matrix (array): predefined weights of the embeddings

    Returns:
        model (tf.keras Model): the sentiment classifier model
    """
    ### START CODE HERE

    lstm_dim = 64
    lstm1_dim = 32
    lstm2_dim = 64
    dense_dim = 6
    filters = 128
    kernel_size = 5

    model = tf.keras.Sequential([
        # This is how you need to set the Embedding layer when using pre-trained embeddings
        tf.keras.layers.Embedding(vocab_size + 1, embedding_dim, input_length=maxlen, weights=[embeddings_matrix], trainable=False),
        #LSTM single
        # tf.keras.layers.LSTM(units=lstm_dim),

        # LSTM bi-directional --this is good... val loss does not go up even line
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_dim)),

        # LSTM RNN -- ok not as good as bi-directional bit flaky at end
        # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm1_dim, return_sequences=True)),
        # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm2_dim)),

        # 1D Conv1D -- ok not as good as bi but very fast not as good as LSTM RNN
        # tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'),
        # tf.keras.layers.GlobalMaxPooling1D(),

        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(dense_dim, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    from tensorflow.keras.optimizers.legacy import Adam

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['accuracy'])



    ### END CODE HERE

    return model

# grader-required-cell

# Create your untrained model
model = create_model(VOCAB_SIZE, EMBEDDING_DIM, MAXLEN, EMBEDDINGS_MATRIX)

# Train the model and save the training history
history = model.fit(train_pad_trunc_seq, train_labels, epochs=20, validation_data=(val_pad_trunc_seq, val_labels))

# grader-required-cell

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = [*range(20)]

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r')
plt.plot(epochs, val_loss, 'b')
plt.title('Training and validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Loss", "Validation Loss"])
plt.show()
