import pandas as pd
import numpy as np
import tensorflow as tf
import argparse
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import mean_squared_error
import keras
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# TODO
# 1) logger
# 2) save model
# 3) validation
# 4) grid search
# 5) experiment
# 6) abeition

def main():

    # get parameters
    parser = argparse.ArgumentParser('Amazon - House Price Prediction')
    parser.add_argument('--dir', type=str, help="Directory to dataset", required=True) #"data/pp-complete.csv"
    parser.add_argument('--dim', type=int, help="Dimension of hidden layer.", default=32)
    parser.add_argument('--epoch', type=int, help="Total number of training epochs to perform.", default=32)
    parser.add_argument("--seed", type=int, help="Random seed for initialization", default=2020)

    args = parser.parse_args()

    # fix random seed for reproducibility
    set_seed(args)

    # Load Data
    x_train, y_train, x_val, y_val, x_test, y_test = load_data(args)

    FEATURE_NUMBER = x_train.shape[-1]

    # Genearte a simple neural network model
    model = generate_model(args, FEATURE_NUMBER)
    model.fit(x_train, y_train, batch_size=32)
    results = model.evaluate(x_test, y_test, batch_size=128, verbose=0)
    print("mse:", results)
        #y_pred = model.predict(x_train)
        #print('Mean squared error: %.2f'
        #      % mean_squared_error(y_train, y_pred))


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





