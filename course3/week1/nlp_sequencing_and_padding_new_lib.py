import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define input sentences
sentences = [
    'i love my dog',
    'I love my cat'
    ]

# Initialize the Tokenizer class
MAX_SEQUENCE_LENGTH = 100

vectorizer = TextVectorization(max_tokens=MAX_SEQUENCE_LENGTH, output_mode='int', output_sequence_length=10)


# Adapt the TextVectorization layer to the training data
vectorizer.adapt(sentences)

# Vectorize a batch of sentences
vectorized_sentences = vectorizer(sentences)

# Print the vocabulary and vectorized sentences
print("Vocabulary:")
vocabulary = vectorizer.get_vocabulary()
print(vocabulary)

word_index = {word: index for index, word in enumerate(vocabulary)}

print("Word Index:")
print(word_index)

padded_sentences = pad_sequences(vectorized_sentences, maxlen=10, padding='post')

# Print the result
print("\nPadded Sequences:")
print(padded_sentences)