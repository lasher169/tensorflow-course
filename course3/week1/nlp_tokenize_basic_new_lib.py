import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

# Define input sentences
sentences = [
    'i love my dog',
    'I love my cat'
    ]

# Initialize the Tokenizer class
MAX_SEQUENCE_LENGTH = 100

# vectorizer = TextVectorization(max_tokens=MAX_SEQUENCE_LENGTH, output_mode='int', output_sequence_length=10)
vectorizer = TextVectorization( output_mode='int', output_sequence_length=10)


# Adapt the TextVectorization layer to the training data
vectorizer.adapt(sentences)

# Vectorize a batch of sentences
vectorized_sentences = vectorizer(sentences)

# Print the vocabulary and vectorized sentences
print("Vocabulary:")
vocabulary = vectorizer.get_vocabulary()
print(vocabulary)

word_index = {index: word for index, word in enumerate(vocabulary)}

print("Word Index:")
print(word_index)


# print("\nVectorized Sentences:")
# print(vectorized_sentences.numpy())