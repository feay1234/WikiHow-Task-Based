import argparse

import pandas as pd
import numpy as np
import random

from keras.callbacks import CSVLogger

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

    all_categories = "Arts and Entertainment·Cars & Other Vehicles·Computers and Electronics·Education and Communications·Family Life·Finance and Business·Food and Entertaining·Health·Hobbies and Crafts·Holidays and Traditions·Home and Garden·Personal Care and Style·Pets and Animals·Philosophy and Religion·Relationships·Sports and Fitness·Travel·Work World·Youth"
    all_categories = all_categories.lower()
    all_categories = all_categories.split("·")

    for category in all_categories:
        category = category.replace(" ", "_")

        MAX_NUM_WORDS = 100000

        try:
            data = pd.read_csv(path+"data/query_task_prediction/same_task/%s.csv" % category, names=["t", "q", "label"])
        except:
            continue

        data["t"] = data["t"].astype(str)
        data["q"] = data["q"].astype(str)

        corpus = data.t.unique().tolist() + data.q.unique().tolist()

        # finally, vectorize the text samples into a 2D integer tensor
        tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
        tokenizer.fit_on_texts(corpus)

        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))


        max_words = len(word_index)
        embedding_layer = get_pretrain_embeddings(path, word_index)

        MAX_SEQUENCE_LENGTH = max([len(i.split()) for i in corpus])
        x_task = pad_sequences(tokenizer.texts_to_sequences(data["t"].tolist()), maxlen=MAX_SEQUENCE_LENGTH)
        x_query = pad_sequences(tokenizer.texts_to_sequences(data["q"].tolist()), maxlen=MAX_SEQUENCE_LENGTH)
        y = data.label.values

        model = getDSSM(embedding_layer, MAX_SEQUENCE_LENGTH)

        csv_logger = CSVLogger(path+'log/query_task_prediction/same_task/%s.out' % category)

        model.fit([x_query, x_task], y, batch_size=128, validation_split=0.4, epoc=5, callbacks=[csv_logger])