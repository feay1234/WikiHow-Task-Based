from keras import Input, Model
from keras.layers import Bidirectional, LSTM, Dense, TimeDistributed

from utils import *

path = "/Users/jarana/workspace/WikiHow-Task-Based/"

x_train, y_train, x_test, y_test, word_index, max_words, maxlen = getAOL()
embedding_layer = get_pretrain_embeddings(path, max_words, word_index)

# maxlen = 10
dim = 100
encoder_inp = Input(shape=(maxlen,))
decoder_inp = Input(shape=(dim,))

embed_end_input = embedding_layer(encoder_inp)

encoder_outputs, state_h, state_c = LSTM(dim, return_state=True)(embed_end_input)
encoder_states = [state_h, state_c]



decoder_inp = Input(shape=(maxlen,))
embed_dec_input = embedding_layer(decoder_inp)

decoder_lstm = LSTM(dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(embed_dec_input, initial_state=encoder_states)
decoder_dense = Dense(max_words, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)


model = Model([encoder_inp, decoder_inp], decoder_outputs)
model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["acc"])

model.fit([x_train, y_train])


def seq2seq_model_builder(embedding_layer, maxlen, max_words, dim=300):
    encoder_inputs = Input(shape=(maxlen,), dtype='int32', )
    encoder_embedding = embedding_layer(encoder_inputs)
    encoder_LSTM = LSTM(dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder_LSTM(encoder_embedding)

    decoder_inputs = Input(shape=(maxlen,), dtype='int32', )
    decoder_embedding = embedding_layer(decoder_inputs)
    decoder_LSTM = LSTM(dim, return_state=True, return_sequences=True)
    decoder_outputs, _, _ = decoder_LSTM(decoder_embedding, initial_state=[state_h, state_c])

    # dense_layer = Dense(VOCAB_SIZE, activation='softmax')
    outputs = TimeDistributed(Dense(max_words, activation='softmax'))(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], outputs)

    return model