import pandas as pd
import numpy as np
import random

from Models import getDSSM
from utils import *

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
MAX_NUM_WORDS = 100000

df = pd.read_csv("data/d3_wikihow.csv")
# corpus = howto.Query.tolist() + df.Query.tolist()
corpus = df.Query.tolist()

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(corpus)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


max_words = len(word_index)
path = ""
embedding_layer = get_pretrain_embeddings(path, word_index)


data = pd.read_csv("data/query_task_prediction.csv", sep="\t", names=["query", "task", "label"])

MAX_SEQUENCE_LENGTH = max([len(i.split()) for i in corpus])
x_query = pad_sequences(tokenizer.texts_to_sequences(data["query"].tolist()), maxlen=MAX_SEQUENCE_LENGTH)
x_task = pad_sequences(tokenizer.texts_to_sequences(data["task"].tolist()), maxlen=MAX_SEQUENCE_LENGTH)
y = data.lavel.values


model = getDSSM(embedding_layer, MAX_SEQUENCE_LENGTH)

model.fit([x_query, x_task], y, batch_size=128, validation_split=0.4, epochs=10)