# grader-required-cell

import csv
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np

with open("./data/NLP_remove_stopwords/bbc-text.csv", 'r') as csvfile:
    print(f"First line (header) looks like this:\n\n{csvfile.readline()}")
    print(f"Each data point looks like this:\n\n{csvfile.readline()}")


# grader-required-cell

# GRADED FUNCTION: remove_stopwords
def remove_stopwords(sentence):
    """
    Removes a list of stopwords

    Args:
        sentence (string): sentence to remove the stopwords from

    Returns:
        sentence (string): lowercase sentence without the stopwords
    """
    # List of stopwords
    stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
                 "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did",
                 "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
                 "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself",
                 "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's",
                 "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only",
                 "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd",
                 "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs",
                 "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
                 "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we",
                 "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's",
                 "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll",
                 "you're", "you've", "your", "yours", "yourself", "yourselves"]

    # Sentence converted to lowercase-only
    sentence = sentence.lower()

    ### START CODE HERE

    words = sentence.split()
    filtered_words = [word for word in words if word not in stopwords]

    sentence = ' '.join(filtered_words)

    ### END CODE HERE
    return sentence

# grader-required-cell

# Test your function
remove_stopwords("I am about to go to the store and get any snack")


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
        next(reader)

        for row in reader:
            labels.append(row[0])
            sentences.append(remove_stopwords(row[1]))

        ### END CODE HERE
    return sentences, labels

# grader-required-cell

# Test your function

# With original dataset
sentences, labels = parse_data_from_file("./data/NLP_remove_stopwords/bbc-text.csv")

print("ORIGINAL DATASET:\n")
print(f"There are {len(sentences)} sentences in the dataset.\n")
print(f"First sentence has {len(sentences[0].split())} words (after removing stopwords).\n")
print(f"There are {len(labels)} labels in the dataset.\n")
print(f"The first 5 labels are {labels[:5]}\n\n")

# With a miniature version of the dataset that contains only first 5 rows
mini_sentences, mini_labels = parse_data_from_file("./data/NLP_remove_stopwords/bbc-text-minimal.csv")

print("MINIATURE DATASET:\n")
print(f"There are {len(mini_sentences)} sentences in the miniature dataset.\n")
print(f"First sentence has {len(mini_sentences[0].split())} words (after removing stopwords).\n")
print(f"There are {len(mini_labels)} labels in the miniature dataset.\n")
print(f"The first 5 labels are {mini_labels[:5]}")


# grader-required-cell

# GRADED FUNCTION: fit_tokenizer
def fit_tokenizer(sentences):
    """
    Instantiates the Tokenizer class

    Args:
        sentences (list): lower-cased sentences without stopwords

    Returns:
        tokenizer (object): an instance of the Tokenizer class containing the word-index dictionary
    """
    ### START CODE HERE
    # Initialize the Tokenizer class
    MAX_SEQUENCE_LENGTH = 100

    vectorizer = TextVectorization(max_tokens=MAX_SEQUENCE_LENGTH, output_mode='int', output_sequence_length=10)

    # Adapt the TextVectorization layer to the training data
    vectorizer.adapt(sentences)

    ### END CODE HERE
    return vectorizer

# grader-required-cell

vectorizer = fit_tokenizer(sentences)

print("Vocabulary:")
vocabulary = vectorizer.get_vocabulary()
print(vocabulary)

# Vectorize a batch of sentences
vectorized_sentences = vectorizer(sentences)

word_index = {index: word for index, word in enumerate(vocabulary)}

print(f"Vocabulary contains {len(word_index)} words\n")

print("<OOV> token included in vocabulary" if "<OOV>" in word_index else "<OOV> token NOT included in vocabulary")


# grader-required-cell

# GRADED FUNCTION: get_padded_sequences
def get_padded_sequences(tokenizer, sentences):
    """
    Generates an array of token sequences and pads them to the same length

    Args:
        tokenizer (object): Tokenizer instance containing the word-index dictionary
        sentences (list of string): list of sentences to tokenize and pad

    Returns:
        padded_sequences (array of int): tokenized sentences padded to the same length
    """

    ### START CODE HERE

    # Pad the sequences using the post padding strategy
    padded_sentences = pad_sequences(vectorized_sentences, maxlen=10, padding='post')
    ### END CODE HERE

    return padded_sentences

# grader-required-cell

padded_sequences = get_padded_sequences(vectorized_sentences, sentences)
print(f"First padded sequence looks like this: \n\n{padded_sequences[0]}\n")
print(f"Numpy array of all sequences has shape: {padded_sequences.shape}\n")
print(f"This means there are {padded_sequences.shape[0]} sequences in total and each one has a size of {padded_sequences.shape[1]}")


# grader-required-cell

# GRADED FUNCTION: tokenize_labels
def tokenize_labels(labels):
    """
    Tokenizes the labels

    Args:
        labels (list of string): labels to tokenize

    Returns:
        label_sequences, label_word_index (list of string, dictionary): tokenized labels and the word-index
    """
    ### START CODE HERE

    # Instantiate the Tokenizer class
    # No need to pass additional arguments since you will be tokenizing the labels
    MAX_SEQUENCE_LENGTH = 100

    processed_labels = [label.replace('\n', '') for label in labels]

    label_vectorizer = TextVectorization(output_mode='int')

    # Fit the tokenizer to the labels
    label_vectorizer.adapt(processed_labels)

    #this is same as texts_to_sequence
    # label_sequences = label_tokenizer.texts_to_sequences(labels)
    label_sequences = label_vectorizer(np.array(processed_labels))

    label_vocabulary = label_vectorizer.get_vocabulary()


    # Save the word index
    label_word_index = {word: index for index, word in enumerate(label_vocabulary)}


    ### END CODE HERE

    return label_sequences, label_word_index

# grader-required-cell

unique_sentences = set(labels)

for sentence in unique_sentences:
    print(sentence)

label_sequences, label_word_index = tokenize_labels(labels)

extracted_sequences = np.array([])

for label_sequence in label_sequences:
    value = label_sequence.numpy()
    # print(label_sequence, type(label_sequence))
    extracted_sequences = np.append(extracted_sequences, value)

extracted_sequences = extracted_sequences.reshape(-1, 1)

print(f"Vocabulary of labels looks like this {label_word_index}\n")


print(f"First ten sequences {np.array(extracted_sequences)[:10]}\n", end=' ')