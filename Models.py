from keras.models import Sequential
from keras import Input, Model
from keras.layers import GlobalMaxPooling1D, Dot


def getDSSM(embedding_layer, MAX_SEQUENCE_LENGTH):

    q_inp = Input(shape=(MAX_SEQUENCE_LENGTH,))
    t_inp = Input(shape=(MAX_SEQUENCE_LENGTH,))
    q_emb = GlobalMaxPooling1D()(embedding_layer(q_inp))
    t_emb = GlobalMaxPooling1D()(embedding_layer(t_inp))
    pred = Dot(-1, normalize=True)([q_emb, t_emb])
    model = Model([q_inp, t_inp], pred)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

