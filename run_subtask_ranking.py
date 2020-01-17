import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import argparse

import pandas as pd
import numpy as np
import random

from keras.callbacks import CSVLogger

from Models import getDSSM, getRanker
from utils import *

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def parse_args():
    parser = argparse.ArgumentParser(description="Run Task Prediction Experiments")

    parser.add_argument('--path', type=str, help='Path to data', default="/Users/jarana/workspace/WikiHow-Task-Based/")
    parser.add_argument('--type', type=str, help='Subtasks or Questions', default="questions")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    path = args.path
    dataTyep = args.type

    all_categories = "Arts and Entertainment·Cars & Other Vehicles·Computers and Electronics·Education and Communications·Family Life·Finance and Business·Food and Entertaining·Health·Hobbies and Crafts·Holidays and Traditions·Home and Garden·Personal Care and Style·Pets and Animals·Philosophy and Religion·Relationships·Sports and Fitness·Travel·Work World·Youth"
    all_categories = all_categories.lower()
    all_categories = all_categories.split("·")

    for category in all_categories:
        category = category.replace(" ", "_")

        MAX_NUM_WORDS = 100000


        try:
            data = pd.read_csv(path+"data/ranking/%s/%s.csv" % (dataTyep, category), names=["t", "q1", "q2", "label1", "label2"])

        except:
            continue

        data["t"] = data["t"].astype(str)
        data["q1"] = data["q1"].astype(str)
        data["q2"] = data["q2"].astype(str)

        corpus = data.t.unique().tolist() + data.q1.unique().tolist() + data.q2.unique().tolist()

        # finally, vectorize the text samples into a 2D integer tensor
        tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
        tokenizer.fit_on_texts(corpus)

        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))


        max_words = len(word_index)
        embedding_layer = get_pretrain_embeddings(path, word_index)

        MAX_SEQUENCE_LENGTH = max([len(i.split()) for i in corpus])
        x_task = pad_sequences(tokenizer.texts_to_sequences(data["t"].tolist()), maxlen=MAX_SEQUENCE_LENGTH)
        x_query1 = pad_sequences(tokenizer.texts_to_sequences(data["q1"].tolist()), maxlen=MAX_SEQUENCE_LENGTH)
        x_query2 = pad_sequences(tokenizer.texts_to_sequences(data["q2"].tolist()), maxlen=MAX_SEQUENCE_LENGTH)
        y = np.array([[i,j] for i,j in zip(data.label1.values, data.label2.values)])

        print(x_task)
        print(x_query1)
        print(x_query2)
        print(y)


        model = getRanker(embedding_layer, MAX_SEQUENCE_LENGTH)

        csv_logger = CSVLogger(path+'log/ranking/%s/%s.out' % (dataTyep,category))

        model.fit([x_task, x_query1, x_query2], y, batch_size=128, validation_split=0.4, epochs=5, callbacks=[csv_logger])

