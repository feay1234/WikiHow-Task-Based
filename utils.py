import numpy as np
import os

from keras.initializers import Constant
from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer


def get_pretrain_embeddings(path, MAX_NUM_WORDS, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, word_index):
    BASE_DIR = path + 'data/'
    GLOVE_DIR = os.path.join(BASE_DIR, 'w2v')
    print('Indexing word vectors.')

    embeddings_index = {}
    with open(os.path.join(GLOVE_DIR, 'glove.840B.300d.txt'), encoding="utf-8") as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embeddings_index[word] = coefs
            # break
    #
    print('Found %s word vectors.' % len(embeddings_index))

    print('Preparing embedding matrix.')

    # prepare embedding matrix
    num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    found = 0
    for word, i in word_index.items():
        if i > MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            if embedding_vector.shape[0] == 0:
                continue
            embedding_matrix[i] = embedding_vector
            found += 1

    print("Token num: %d, Found Tokens: %d" % (len(word_index), found))

    # load pre-trained word embeddings into an Embedding layer
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                embeddings_initializer=Constant(embedding_matrix))

    return embedding_layer




def save2file(path, output):
    with open(path, "a") as myfile:
        myfile.write(output + "\n")
    print(output)


def getAOL(dir="data/tmp/", MAX_NUM_WORDS=50000, MAX_SEQUENCE_LENGTH=50):

    q1, q2 = [], []
    for i in os.listdir(dir):
        f = open("%s%s" % (dir, i), "r")
        q = f.readline().strip().split("\t")
        for i in range(len(q) - 1):
            q1.append(q[i])
            q2.append(q[i + 1])

    corpus = q1 + q2

    # finally, vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(corpus)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    # Update max length
    MAX_SEQUENCE_LENGTH = min(np.max([len(i) for i in corpus]), MAX_SEQUENCE_LENGTH)
    print("Updated maxlen: %d" % MAX_SEQUENCE_LENGTH)

    x1 = tokenizer.texts_to_sequences(q1)
    x2 = tokenizer.texts_to_sequences(q2)

    return x1, x2

data = getAOL()