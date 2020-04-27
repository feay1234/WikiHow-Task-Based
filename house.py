import pandas as pd
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


if __name__ == "__main__":

    SELECT_COLUMN = [1,2,4,6,11] # We only consider price, date of the purchase, property type, lease duration and town
    df = pd.read_csv("data/pp-complete.csv", names=["price", "date", "ptype", "duration", "town"], sep=",", usecols=SELECT_COLUMN)

    for c in ['duration', 'ptype']:
        df[c] = df[c].astype('category')
        df[c] = df[c].cat.codes

    df['isLondon'] = [1 if i == "LONDON" else 0 for i in df['town']]
    df['date'] =  pd.to_datetime(df['date'], format='%Y-%m-%d')
    df["year"] = df['date'].dt.year

    # Split the data into training/testing sets
    df_train = df[df.date < "2016-1-1"]
    df_test = df[df.date >= "2016-1-1"]
    #df_test = df_test[df.date < "2017-1-1"]
    x_train = df_train[['year', 'ptype', 'duration', 'isLondon']].values
    y_train = df_train['price'].values

    x_test = df_test[['year', 'ptype', 'duration', 'isLondon']].values
    y_test = df_test['price'].values


    # define the model
    def baseline():
        # create model
        model = Sequential()
        model.add(Dense(4, input_dim=4, activation='relu', kernel_initializer='normal'))
        model.add(Dense(1, kernel_initializer='normal'))
        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model
    model = baseline()
    for i in range(3):
        model.fit(x_train, y_train, batch_size=32)
        results = model.evaluate(x_test, y_test, batch_size=128, verbose=0)
        print("mse:", results)
        #y_pred = model.predict(x_train)
        #print('Mean squared error: %.2f'
        #      % mean_squared_error(y_train, y_pred))
