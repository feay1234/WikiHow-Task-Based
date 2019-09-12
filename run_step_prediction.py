import argparse

import pandas as pd
import numpy as np
import random

from Models import getDSSM, getRanker
from utils import *

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def parse_args():
    parser = argparse.ArgumentParser(description="Run Task Prediction Experiments")

    parser.add_argument('--path', type=str, help='Path to data', default="/Users/jarana/workspace/WikiHow-Task-Based/")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    path = args.path


    MAX_NUM_WORDS = 100000


    data = pd.read_csv(path+"data/step_prediction.csv", sep="\t", names=["s1", "s2", "s1_text", "s2_text", "task"])

    # corpus = howto.Query.tolist() + df.Query.tolist()
    corpus = list(set(data.s1.tolist() + data.s2.tolist() + data.task.tolist()))
    # corpus = [str(i) for i in corpus]
    # finally, vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(corpus)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))


    max_words = len(word_index)
    embedding_layer = get_pretrain_embeddings(path, word_index)

    s1 = []
    s2 = []
    label = []
    for idx, row in data.iterrows():
        rand = np.random.randint(2)
        if rand == 0:
            s1.append(row.s1)
            s2.append(row.s2)
        else:
            s1.append(row.s2)
            s2.append(row.s1)
        y = np.zeros(2)
        y[rand] = 1
        label.append(y)

    label = np.array(label)


    MAX_SEQUENCE_LENGTH = min(max([len(i.split()) for i in corpus]), 200)
    x_task = pad_sequences(tokenizer.texts_to_sequences(data["task"].tolist()), maxlen=MAX_SEQUENCE_LENGTH)
    x_step1 = pad_sequences(tokenizer.texts_to_sequences(s1), maxlen=MAX_SEQUENCE_LENGTH)
    x_step2 = pad_sequences(tokenizer.texts_to_sequences(s2), maxlen=MAX_SEQUENCE_LENGTH)


    model = getRanker(embedding_layer, MAX_SEQUENCE_LENGTH)

<<<<<<< HEAD
    model.fit([x_task, x_step1, x_step2], y, batch_size=128, validation_split=0.3, epochs=10, verbose=2)
=======
    model.fit([x_task, x_step1, x_step2], label, batch_size=128, validation_split=0.3, epochs=10, verbose=2)
>>>>>>> c332c4a27a589db7030088fdc4ce3194f7594093
