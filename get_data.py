import pickle
import numpy as np
import tensorflow as tf


# Reads pickled lists containing cleaned data (stored as lists of tokens)
def read_data():
    with open('./input/data_pickles/train_list_sentence1.txt', "rb") as file:
        train_list_sentence1 = pickle.load(file)

    with open('./input/data_pickles/train_list_sentence2.txt', "rb") as file:
        train_list_sentence2 = pickle.load(file)

    with open('./input/data_pickles/train_list_gold_label.txt', "rb") as file:
        train_list_gold_label = pickle.load(file)

    with open('./input/data_pickles/test_list_sentence1.txt', "rb") as file:
        test_list_sentence1 = pickle.load(file)

    with open('./input/data_pickles/test_list_sentence2.txt', "rb") as file:
        test_list_sentence2 = pickle.load(file)

    with open('./input/data_pickles/test_list_gold_label.txt', "rb") as file:
        test_list_gold_label = pickle.load(file)

    data = [[train_list_sentence1], [train_list_sentence2], [train_list_gold_label],
            [test_list_sentence1], [test_list_sentence2], [test_list_gold_label]]

    return data


# Converts data to required format
def convert_data(data):
    # Get the sentences and labels from composite data
    list_sentence1 = data[0][0]
    list_sentence2 = data[1][0]
    list_gold_label = data[2][0]

    # Merge each sublist (tokens list of each sentence) to a string
    corpus_sentence1 = [' '.join(item) for item in list_sentence1]
    corpus_sentence2 = [' '.join(item) for item in list_sentence2]
    num_samples = len(list_gold_label)

    # Create a composite corpus over which to train the Keras tokenizer
    # Corresponding lines of sentence1 and sentence2 are merged together
    corpus = [corpus_sentence1[ind] + " " + corpus_sentence2[ind] for ind in range(num_samples)]

    # There are entries in dataset without any gold_label (with gold_label entry as "-")
    # I choose to delete those from my data as they do not provide any information
    del_list = []
    labels = [None] * num_samples
    for ind, item in enumerate(list_gold_label):
        if item == "contradiction":
            labels[ind] = 0
        elif item == "neutral":
            labels[ind] = 1
        elif item == "entailment":
            labels[ind] = 2
        else:
            labels[ind] = 99
            del_list.append(ind)

    # Delete entries with gold_label "-"
    del_list.sort(reverse=True)
    for ind in del_list:
        del corpus[ind]
        del corpus_sentence1[ind]
        del corpus_sentence2[ind]
        del labels[ind]

    labels = np.array(labels)
    data_converted = [corpus_sentence1, corpus_sentence2, labels]

    # """
    print(f"Non labelled: {len(del_list)}")
    print(f"Contradiction: {np.sum(labels == 0)}")
    print(f"Neutral: {np.sum(labels == 1)}")
    print(f"Entailment: {np.sum(labels == 2)}")
    # """

    return data_converted, corpus


def glove_dict(EMBEDDING_DIM):
    embedding_dict = {}

    # Open the GloVe embedding file
    glove_dir = './input/embeddings/glove.840B.'+ str(EMBEDDING_DIM)+'d.txt'
    file = open(glove_dir, encoding="utf8")

    for line in file:
        # Spilt the word and its embedding vector
        line_list = line.split(' ')
        word = line_list[0]
        embeddings = np.asarray(line_list[1:], dtype='float32')

        # Store the word and its embedding vector in a dictionary
        embedding_dict[word] = embeddings

    file.close()

    # Store the dictionary as a pickle file to reduce thw overhead of loading
    with open(f'./input/embeddings/glove_dict_840_{EMBEDDING_DIM}d.pickle', "wb") as file:
        pickle.dump(embedding_dict, file)

    return embedding_dict


def embedding_matrix(corpus, EMBEDDING_DIM, VOCAB_SIZE):
    # Try to load GloVe embedding dictionary if it exists. If not, create one
    try:
        with open(f'./input/embeddings/glove_dict_840_{EMBEDDING_DIM}d.pickle', "rb") as file:
            glove_embedding = pickle.load(file)
    except:
        glove_embedding = glove_dict(EMBEDDING_DIM)

    # Initialize and fit Keras tokenizer to convert words to integers
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(corpus)

    # Save the tokenizer as a pickle file so that the same tokenizer (word-integer)
    # mapping can be used during testing time
    with open('./model/tokenizer.pickle', "wb") as file:
        pickle.dump(tokenizer, file)

    # Get an word-integer dictionary and use that to create an weight matrix
    # i-th column of weight matrix will have the vector of word with integer value i in dictionary
    word_index = tokenizer.word_index
    embed_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))

    for word, ind in word_index.items():
        # Get the embedding vector from GloVe dictionary, if available
        # Words not in the Glove would have the embedding matrix vector as random values uniformly generated between -0.05 and 0.05
        embedding_vector = glove_embedding.get(word)

        if embedding_vector is not None:
            embed_matrix[ind] = embedding_vector
        else:
            embed_matrix[ind] = np.array([np.random.uniform(-0.05, 0.05) for _ in range(EMBEDDING_DIM)])

    return embed_matrix, tokenizer


def preprocess_traindata(train_data, MAX_SEQ_LEN, EMBEDDING_DIM, VOCAB_SIZE):
    # print("Train Data information\n")

    # Convert data to required format
    data, corpus = convert_data(train_data)

    # Obtain the embedding weight matrix and the tokenizer
    embed_matrix, tokenizer = embedding_matrix(corpus, EMBEDDING_DIM, VOCAB_SIZE)

    # Process the data to integer sequences and labels to one-hot labels
    sequence = lambda sentence: tf.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(sentence), maxlen=MAX_SEQ_LEN)
    process = lambda item: (sequence(item[0]), sequence(item[1]), tf.keras.utils.to_categorical(item[2]))

    training_data = process(data)

    return training_data, embed_matrix

# Function to preprocess test data
def preprocess_testdata(test_data, MAX_SEQ_LEN):
    # print("Test Data information\n")

    # Convert data to required format
    data, _ = convert_data(test_data)

    # Load the tokenizer from pickle file
    with open('./model/tokenizer.pickle', "rb") as file:
        tokenizer = pickle.load(file)

    # Process the data to integer sequences and labels to one-hot labels
    sequence = lambda sentence: tf.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(sentence), maxlen=MAX_SEQ_LEN)
    process = lambda item: (sequence(item[0]), sequence(item[1]), tf.keras.utils.to_categorical(item[2]))

    test_data = process(data)

    return test_data
