from time import strftime, localtime
import pandas as pd
import numpy as np
import tensorflow as tf
import argparse
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

def main():

    # get parameters
    parser = argparse.ArgumentParser('Amazon - House Price Prediction')
    parser.add_argument('--dir', type=str, help="Directory to dataset", required=True, default="data/pp-complete.csv") #"data/pp-complete.csv"
    parser.add_argument('--dim', type=int, help="Dimension of hidden layer.", default=32)
    parser.add_argument('--epoch', type=int, help="Total number of training epochs to perform.", default=100)
    parser.add_argument("--seed", type=int, help="Random seed for initialization", default=2020)

    args = parser.parse_args()

    timestamp = strftime('%Y_%m_%d_%H_%M_%S', localtime())
    model_name = "house_nn_d%d_%s" % (args.dim, timestamp)

    # fix random seed for reproducibility
    set_seed(args)

    # Load Data
    x_train, y_train, x_val, y_val, x_test, y_test = load_data(args)

    FEATURE_NUMBER = x_train.shape[-1]

    # Genearte a simple neural network model
    model = generate_model(args, FEATURE_NUMBER)

    # Generate Keras callbacks for logging, early stopping and saving the model's parameters
    csv_logger = CSVLogger('%s.log' % model_name)
    early_stop = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
    checkpointer = ModelCheckpoint(filepath='%s.hdf5' % model_name, verbose=1, save_best_only=True)

    # Train the model and validate the model on the validation set.
    # Early Stopping: the model will stop training if the loss on the validation increases
    model.fit(x_train[:10], y_train[:10], batch_size=128, epochs=args.epoch, verbose=1,
              validation_data=(x_val[:10], y_val[:10]), callbacks=[checkpointer, csv_logger, early_stop])

    # Evaluate the model on the test set.
    print('\n# Evaluate on test data')
    results = model.evaluate(x_test[:10], y_test[:10], batch_size=128)
    print('test loss, test acc:', results)


def generate_model(args, feature_number):
    """
    Generate a single-layer neural network model for regression task.
    """
    model = Sequential()
    model.add(Dense(args.dim, input_dim=feature_number, activation='relu', kernel_initializer='normal'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def preprocessing(df):
    """
    Perform a simple preprocessing on the dataset
    """
    # remove rows with missing values
    df = df.dropna()
    # remove duplicate rows
    df = df.drop_duplicates()
    # remove properties that cost less than £100
    df = df[df.price > 100]

    return df

def feature_exaction(df):
    # Generate a binary feature that indicates whether or not the property is in London.
    df['isLondon'] = [1 if i == "LONDON" else 0 for i in df['town']]
    return df

def load_data(args):
    """
    Read the dataset, perform data prepropessing, extract feature
    """

    # We only consider price, date of the purchase, property type, lease duration and town.
    SELECT_COLUMNS = [1,2,4,6,11]
    # Categorical features that we use to train model
    CATEGORY_FEATURES = ['propertyType', 'duration', 'isLondon']

    # Read data from a given file
    df = pd.read_csv(args.dir, names=["price", "date", "propertyType", "duration", "town"], sep=",", usecols=SELECT_COLUMNS)

    # Convert string to datetime
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

    # Data preprocessing
    df = preprocessing(df)

    # Feature Extraction
    df = feature_exaction(df)

    # Split the data into train, validation and test sets
    df_train = df[df.date < "2015-1-1"]
    df_val = df[(df.date >= "2015-1-1") & (df.date < "2016-1-1")]
    df_test = df[df.date >= "2016-1-1"]

    # Create one-hot encoder and train it on the dataset
    encoder = OneHotEncoder()
    encoder.fit(df[CATEGORY_FEATURES])

    # We convert categorical features into one-hot features
    x_train, y_train = encoder.transform(df_train[CATEGORY_FEATURES].values).toarray()
    x_val = encoder.transform(df_val[CATEGORY_FEATURES].values).toarray()
    x_test = encoder.transform(df_test[CATEGORY_FEATURES].values).toarray()

    # We use property's price as ground truth
    y_train = df_train['price'].values
    y_val = df_val['price'].values
    y_test = df_test['price'].values

    # check consistency of feature number across the train, val and test sets.
    assert x_train.shape[-1] == x_val.shape[-1] == x_test.shape[-1]

    return x_train, y_train, x_val, y_val, x_test, y_test

def set_seed(args):
    """
    Set the random seed.
    """
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

if __name__ == "__main__":
    main()





