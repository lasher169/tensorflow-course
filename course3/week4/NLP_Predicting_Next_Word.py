# grader-required-cell

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, GRU
import os
import requests
import shutil

url = "https://docs.google.com/uc?export=download&confirm=t&id=108jAePKK4R3BVYBbYJZ32JWUwxeMg20K"

current_file_path = __file__
current_file_name = os.path.splitext(os.path.basename(current_file_path))[0]

dst = "data/"+ current_file_name

def fetch(url, extracted_dst):
    # Send a GET request to the URL
    response = requests.get(url)

    if os.path.exists(extracted_dst):
        shutil.rmtree(extracted_dst)
        print(extracted_dst + " deleted")

    # Ensure the directory exists, create it if necessary
    if not os.path.exists(extracted_dst):
        os.makedirs(extracted_dst)

    filename = os.path.join(extracted_dst, "sonnets.txt")

    # Check if request was successful (status code 200)
    if response.status_code == 200:
        with open(filename, 'wb') as file:
            file.write(response.content)
            print(f"File downloaded to: {filename}")
    else:
        print("Failed to download the zip file.")


fetch(url, dst)

# Read the data
with open(dst+'/sonnets.txt') as f:
    data = f.read()

# Convert to lower case and save as a list
corpus = data.lower().split("\n")

print(f"There are {len(corpus)} lines of sonnets\n")
print(f"The first 5 lines look like this:\n")
for i in range(5):
  print(corpus[i])

  # Read the data
  with open(dst+'/sonnets.txt') as f:
      data = f.read()

  # Convert to lower case and save as a list
  corpus = data.lower().split("\n")

print(f"There are {len(corpus)} lines of sonnets\n")
print(f"The first 5 lines look like this:\n")
for i in range(5):
  print(corpus[i])

# grader-required-cell

# grader-required-cell

tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

corpus[0]

tokenizer.texts_to_sequences(corpus[0])

# grader-required-cell

# GRADED FUNCTION: n_gram_seqs
def n_gram_seqs(corpus, tokenizer):
    """
    Generates a list of n-gram sequences

    Args:
        corpus (list of string): lines of texts to generate n-grams for
        tokenizer (object): an instance of the Tokenizer class containing the word-index dictionary

    Returns:
        input_sequences (list of int): the n-gram sequences for each line in the corpus
    """
    input_sequences = []

    ### START CODE HERE

    # Loop over every line
    for line in corpus:

        # Tokenize the current line
        token_list = tokenizer.texts_to_sequences([line])[0]

        # Loop over the line several times to generate the subphrases
        for i in range(1, len(token_list)):
            # Generate the subphrase
            n_gram_sequence = token_list[:i + 1]

            # Append the subphrase to the sequences list
            input_sequences.append(n_gram_sequence)

    ### END CODE HERE

    return input_sequences

# grader-required-cell

# Test your function with one example
first_example_sequence = n_gram_seqs([corpus[0]], tokenizer)

print("n_gram sequences for first example look like this:\n")
print(first_example_sequence)

# grader-required-cell

# Test your function with a bigger corpus
next_3_examples_sequence = n_gram_seqs(corpus[1:4], tokenizer)

print("n_gram sequences for next 3 examples look like this:\n")
print(next_3_examples_sequence)

# grader-required-cell

# Apply the n_gram_seqs transformation to the whole corpus
input_sequences = n_gram_seqs(corpus, tokenizer)

# Save max length
max_sequence_len = max([len(x) for x in input_sequences])

print(f"n_grams of input_sequences have length: {len(input_sequences)}")
print(f"maximum length of sequences is: {max_sequence_len}")

# grader-required-cell

# GRADED FUNCTION: pad_seqs
def pad_seqs(input_sequences, maxlen):
    """
    Pads tokenized sequences to the same length

    Args:
        input_sequences (list of int): tokenized sequences to pad
        maxlen (int): maximum length of the token sequences

    Returns:
        padded_sequences (array of int): tokenized sequences padded to the same length
    """
    ### START CODE HERE
    padded_sequences = np.array(pad_sequences(input_sequences, maxlen=maxlen, padding='pre'))

    return padded_sequences
    ### END CODE HERE

# grader-required-cell

# Test your function with the n_grams_seq of the first example
first_padded_seq = pad_seqs(first_example_sequence, max([len(x) for x in first_example_sequence]))
print(first_padded_seq)

# grader-required-cell

# Test your function with the n_grams_seq of the next 3 examples
next_3_padded_seq = pad_seqs(next_3_examples_sequence, max([len(s) for s in next_3_examples_sequence]))
next_3_padded_seq

# Pad the whole corpus
input_sequences = pad_seqs(input_sequences, max_sequence_len)

print(f"padded corpus has shape: {input_sequences.shape}")

# grader-required-cell

# GRADED FUNCTION: features_and_labels
# this sets the next word as label and continues on until the whole input is read
def features_and_labels(input_sequences, total_words):
    """
    Generates features and labels from n-grams

    Args:
        input_sequences (list of int): sequences to split features and labels from
        total_words (int): vocabulary size

    Returns:
        features, one_hot_labels (array of int, array of int): arrays of features and one-hot encoded labels
    """
    ### START CODE HERE
    import tensorflow as tf

    #NOTE IMPORTANT!!!!
    features = input_sequences[:, :-1]
    labels = input_sequences[:, -1]
    one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=total_words)
    ### END CODE HERE

    return features, one_hot_labels

# grader-required-cell

# Test your function with the padded n_grams_seq of the first example
first_features, first_labels = features_and_labels(first_padded_seq, total_words)

print(f"labels have shape: {first_labels.shape}")
print("\nfeatures look like this:\n")
first_features

# grader-required-cell

# Split the whole corpus
features, labels = features_and_labels(input_sequences, total_words)

print(f"features have shape: {features.shape}")
print(f"labels have shape: {labels.shape}")

# GRADED FUNCTION: create_model
def create_model(total_words, max_sequence_len):
    """
    Creates a text generator model

    Args:
        total_words (int): size of the vocabulary for the Embedding layer input
        max_sequence_len (int): length of the input sequences

    Returns:
        model (tf.keras Model): the text generator model
    """
    model = Sequential()
    ### START CODE HERE
    model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
    model.add(Bidirectional(LSTM(100, return_sequences=True)))
    model.add(GRU(1024, return_sequences=False))
    model.add(Dense(total_words, activation='softmax'))

    from tensorflow.keras.optimizers.legacy import Adam

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['accuracy'])

    ### END CODE HERE

    return model

# Get the untrained model
model = create_model(total_words, max_sequence_len)

# Train the model
history = model.fit(features, labels, epochs=50, verbose=1)

# Take a look at the training curves of your model

acc = history.history['accuracy']
loss = history.history['loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.title('Training accuracy')

plt.figure()

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.title('Training loss')
plt.legend()

plt.show()


seed_text = "Help me Obi Wan Kenobi, you're my only hope"
next_words = 100

for _ in range(next_words):
    # Convert the text into sequences
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    # Pad the sequences
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    # Get the probabilities of predicting a word
    predicted = model.predict(token_list, verbose=0)
    # Choose the next word based on the maximum probability
    predicted = np.argmax(predicted, axis=-1).item()
    # Get the actual word from the word index
    output_word = tokenizer.index_word[predicted]
    # Append to the current text
    seed_text += " " + output_word

print(seed_text)
