from time import strftime, localtime
import pandas as pd
import numpy as np
import tensorflow as tf
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from tabulate import tabulate

timestamp = strftime('%Y_%m_%d_%H_%M_%S', localtime())

def main():
    # get parameters
    parser = argparse.ArgumentParser('Amazon - House Price Prediction')
    parser.add_argument('--dir', type=str, help="Directory to dataset", required=True)
    parser.add_argument('--dim', type=int, help="Dimension of hidden layer.", default=32)
    parser.add_argument('--epoch', type=int, help="Total number of training epochs to perform.", default=100)
    parser.add_argument("--seed", type=int, help="Random seed for initialization", default=2020)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=256)
    parser.add_argument("--verbose", type=int, help="Verbose (0: disable, 1: enable 2: partially enable)", default=1)
    parser.add_argument("--train_on", type=str, help="Train the model on the houses sold before a particular time", default="2015-1-1")
    parser.add_argument("--test_on", type=str, help="Test the model on the houses sold after a particular time", default="2016-1-1")
    args = parser.parse_args()

    # Model name
    model_name = "house_nn_d%d_%s" % (args.dim, timestamp)

    # fix random seed for reproducibility
    set_seed(args)

    # Load Data
    x_train, y_train, x_val, y_val, x_test, y_test, df_test = load_data(args)

    feature_number = x_train.shape[-1]

    # Genearte a simple neural network model
    model = generate_model(args, feature_number)

    # Generate Keras callbacks for logging, early stopping and saving the model's parameters
    csv_logger = CSVLogger('%s.log' % model_name)
    early_stop = EarlyStopping(monitor='val_loss', patience=2, verbose=args.verbose)
    checkpointer = ModelCheckpoint(filepath='%s.hdf5' % model_name, verbose=args.verbose, save_best_only=True)

    all_results = [] # to save models' results

    # Train and evaluate neural model. We train the model and validate the model on the validation set. Early Stopping is applied and the model will stop training if the loss on the validation increases
    model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epoch, verbose=args.verbose, validation_data=(x_val, y_val), callbacks=[checkpointer, csv_logger, early_stop])
    y_pred = model.predict(x_test, batch_size=args.batch_size)
    results = evaluate(y_test, y_pred, "Neural model")
    all_results.append(results)

    # Train and evaluate unsupervised baselines
    baselines = ['mean_price', 'mean_price_propertyType', 'mean_price_duration', 'mean_price_isLondon']
    for b in baselines:
        y_pred = df_test[b]
        results = evaluate(y_test, y_pred, b)
        all_results.append(results)

    print(tabulate(pd.DataFrame(all_results)[["Model", "MAE", "MALE", "RMSE", "RMSLE"]], headers='keys', tablefmt='psql'))

def evaluate(y_test, y_pred, name):
    """
    Evaluate the model's prediction on the test set using multiple metrices.
    """
    MAE = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))
    MALE = mean_absolute_error(y_test, y_pred)
    RMSE = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_pred)))
    RMSLE = np.sqrt(mean_squared_error(y_test, y_pred))
    return {"Model": name, "MAE": MAE, "MALE": MALE, "RMSE": RMSE, "RMSLE": RMSLE}

def generate_model(args, feature_number):
    """
    Generate a single-layer neural network model for regression task.
    """
    model = Sequential()
    model.add(Dense(args.dim, input_dim=feature_number, activation='relu', kernel_initializer='normal'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def preprocessing(df):
    """
    Perform a simple preprocessing on the dataset
    """
    # Remove rows with missing values
    df = df.dropna()
    # Remove duplicate rows
    df = df.drop_duplicates()
    # Remove properties that cost less than £100
    df = df[df.price > 100]
    # Convert price into log scale
    df['price'] = np.log1p(df['price'])
    return df


def feature_exaction(args, df):
    """
    Generate a binary feature that indicates whether or not the property is in London.
    """
    df['isLondon'] = [1 if i == "LONDON" else 0 for i in df['town']]
    df['year'] = df.date.dt.year

    df_train = df[df.date < args.train_on]

    # Mean-price features
    for c in ["isLondon", 'propertyType', 'duration']:
        meanPrice = df_train.groupby(c)['price'].mean().to_dict()
        df["mean_price_%s" % c] = [meanPrice[i] for i in df[c]]

    df['mean_price'] = [df_train.price.mean()] * len(df)

    return df

def load_data(args):
    """
    Read the dataset, perform data prepropessing, extract feature
    """
    # We only consider price, date of the purchase, property type, lease duration and town.
    SELECT_COLUMNS = [1, 2, 4, 6, 11]
    # Feature name that we use to train our model
    CATEGORY_FEATURES = ['propertyType', 'duration', 'isLondon']
    NUMERICAL_FEATURES = ['year', 'mean_price', 'mean_price_propertyType', 'mean_price_duration', 'mean_price_isLondon']

    # Read data from a given file
    df = pd.read_csv(args.dir, names=["price", "date", "propertyType", "duration", "town"], sep=",", usecols=SELECT_COLUMNS)

    # Convert string to datetime
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

    # Data preprocessing
    df = preprocessing(df)

    # Feature Extraction
    df = feature_exaction(args, df)

    # Split the data into train, validation and test sets
    df_train = df[df.date < args.train_on]
    df_val = df[(df.date >= args.train_on) & (df.date < args.test_on)]
    df_test = df[df.date >= args.test_on]

    # Create one-hot encoder and train it on the dataset
    encoder = OneHotEncoder()
    encoder.fit(df[CATEGORY_FEATURES])

    # We convert categorical features into one-hot features
    x_train = encoder.transform(df_train[CATEGORY_FEATURES].values).toarray()
    x_val = encoder.transform(df_val[CATEGORY_FEATURES].values).toarray()
    x_test = encoder.transform(df_test[CATEGORY_FEATURES].values).toarray()

    # Combine categorical features and hand-crafted features
    x_train = np.concatenate([x_train, df_train[NUMERICAL_FEATURES].values], axis=1)
    x_val = np.concatenate([x_val, df_val[NUMERICAL_FEATURES].values], axis=1)
    x_test = np.concatenate([x_test, df_test[NUMERICAL_FEATURES].values], axis=1)

    # We use property's price as ground truth
    y_train = df_train['price'].values
    y_val = df_val['price'].values
    y_test = df_test['price'].values

    # check consistency of feature number across the train, val and test sets.
    assert x_train.shape[-1] == x_val.shape[-1] == x_test.shape[-1]

    return x_train, y_train, x_val, y_val, x_test, y_test, df_test

def set_seed(args):
    np.random.seed(args.seed)
    tf.compat.v1.set_random_seed(args.seed)

if __name__ == "__main__":
    main()