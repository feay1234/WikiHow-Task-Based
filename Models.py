from keras.models import Sequential
from keras import Input, Model
from keras.layers import GlobalMaxPooling1D, Dot, LSTM, Dense, Concatenate


def getDSSM(embedding_layer, MAX_SEQUENCE_LENGTH):

    q_inp = Input(shape=(MAX_SEQUENCE_LENGTH,))
    t_inp = Input(shape=(MAX_SEQUENCE_LENGTH,))
    lstm = LSTM(100)
    # q_emb = GlobalMaxPooling1D()(embedding_layer(q_inp))
    # t_emb = GlobalMaxPooling1D()(embedding_layer(t_inp))
    q_emb = lstm(embedding_layer(q_inp))
    t_emb = lstm(embedding_layer(t_inp))
    pred = Dot(-1, normalize=True)([q_emb, t_emb])
    model = Model([q_inp, t_inp], pred)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

def getRanker(embedding_layer, MAX_SEQUENCE_LENGTH):
    step1_inp = Input(shape=(MAX_SEQUENCE_LENGTH,))
    step2_inp = Input(shape=(MAX_SEQUENCE_LENGTH,))
    task_inp = Input(shape=(MAX_SEQUENCE_LENGTH,))
    lstm = LSTM(100)
    # q_emb = GlobalMaxPooling1D()(embedding_layer(q_inp))
    t_emb = lstm(embedding_layer(task_inp))
    s1_emb = lstm(embedding_layer(step1_inp))
    s2_emb = lstm(embedding_layer(step2_inp))
    relu = Dense(100, activation='relu')
    dense = Dense(2, activation='softmax')

    pred1 = Dot(-1, normalize=True)([t_emb, s1_emb])
    pred2 = Dot(-1, normalize=True)([t_emb, s2_emb])
    concat = Concatenate(axis=-1)([pred1, pred2])


    # concat = Concatenate(axis=-1)([t_emb, s1_emb, s2_emb])
    # pred = dense(relu(concat))
    pred = dense(concat)
    model = Model([task_inp, step1_inp, step2_inp], pred)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model
