import tensorflow as tf
import os
import requests
import shutil
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
urls = []

# Append elements in the dictionaries into each list
for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [sentence.replace('-', ' ') for sentence in sentences]

sentence = list(set(sentences))

# print(sentences)


# Initialize the Tokenizer class
vectorized = TextVectorization( output_mode='int', output_sequence_length=10, split='whitespace', encoding='utf-8')

# Generate the word index dictionary
vectorized.adapt(sentences)

# Vectorize a batch of sentences
vectorized_sentences = vectorized(sentences)

# Print the length of the word index
# word_index = tokenizer.word_index
vocabulary = vectorized.get_vocabulary()

# print(vocabulary)

word_index = {word: index for index, word in enumerate(vocabulary)}
print(f'number of words in word_index: {len(word_index)}')

# Print the word index
print(f'word_index: {word_index}')
print()
# for word, index in word_index.items():
#     print(f"{word}: {index}")

# Generate and pad the sequences


# Print a sample headline
index = 2
print(f'sample headline: {sentences[index]}')
print(f'padded sequence: {vectorized_sentences[index]}')
print()

# Print dimensions of padded sequences