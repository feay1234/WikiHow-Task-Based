import numpy as np
import os

from keras.initializers import Constant
from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


def get_pretrain_embeddings(path, MAX_NUM_WORDS, word_index, EMBEDDING_DIM=300):
    BASE_DIR = path + 'data/'
    GLOVE_DIR = os.path.join(BASE_DIR, 'w2v')
    print('Indexing word vectors.')

    embeddings_index = {}
    with open(os.path.join(GLOVE_DIR, 'glove.42B.300d.txt'), encoding="utf-8") as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embeddings_index[word] = coefs
            break
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


def tagger(decoder_input_sentence, tag):
    if tag == "<EOS>":
        final_target = [text + tag for text in decoder_input_sentence]
    elif tag == "<SOS>":
        final_target = [tag + text for text in decoder_input_sentence]
    return final_target


def save2file(path, output):
    with open(path, "a") as myfile:
        myfile.write(output + "\n")
    print(output)


def getAOL(dir="data/tmp/", MAX_NUM_WORDS=70000, MAX_SEQUENCE_LENGTH=50):

    encoder_inputs, decoder_inputs = [], []
    for i in os.listdir(dir)[:100]:
        f = open("%s%s" % (dir, i), "r")
        q = f.readline().strip().split("\t")
        for i in range(len(q) - 1):
            encoder_inputs.append(q[i])
            decoder_inputs.append(q[i + 1])

    # add EOS and SOS into decoder
    decoder_inputs = tagger(decoder_inputs)
    print(decoder_inputs)

    corpus = encoder_inputs + decoder_inputs

    # finally, vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(corpus)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    # Update max length
    print(np.max([len(i) for i in corpus]))
    MAX_SEQUENCE_LENGTH = min(np.max([len(i) for i in corpus]), MAX_SEQUENCE_LENGTH)
    print("Updated maxlen: %d" % MAX_SEQUENCE_LENGTH)

    encoder_inputs = pad_sequences(tokenizer.texts_to_sequences(encoder_inputs), maxlen=MAX_SEQUENCE_LENGTH)
    decoder_inputs = pad_sequences(tokenizer.texts_to_sequences(decoder_inputs), maxlen=MAX_SEQUENCE_LENGTH)

    x_train, x_test, y_train, y_test = train_test_split(encoder_inputs, decoder_inputs, test_size=0.33, random_state=2019)

    # decoder_input_data =


    return x_train, x_test, y_train, y_test, word_index, len(word_index), MAX_SEQUENCE_LENGTH

# data = getAOL()
# print(data[0])
# print(data[2])
