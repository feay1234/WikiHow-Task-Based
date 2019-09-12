import argparse

import pandas as pd
import numpy as np
import random

from Models import getDSSM
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


    data = pd.read_csv(path+"data/step_task_prediction.csv", sep="\t", names=["step", "step_desc", "task", "label"])

    # corpus = howto.Query.tolist() + df.Query.tolist()
    corpus = list(set(data.step.tolist() + data.step_desc.tolist() + data.task.tolist()))
    # corpus = [str(i) for i in corpus]
    # finally, vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(corpus)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))


    max_words = len(word_index)
    embedding_layer = get_pretrain_embeddings(path, word_index)



    MAX_SEQUENCE_LENGTH = min(max([len(i.split()) for i in corpus]), 200)
    x_step = pad_sequences(tokenizer.texts_to_sequences(data["step"].tolist()), maxlen=MAX_SEQUENCE_LENGTH)
    x_task = pad_sequences(tokenizer.texts_to_sequences(data["task"].tolist()), maxlen=MAX_SEQUENCE_LENGTH)
    y = data.label.values


    model = getDSSM(embedding_layer, MAX_SEQUENCE_LENGTH)

    model.fit([x_step, x_task], y, batch_size=128, validation_split=0.3, epochs=10, verbose=2)