import pandas as pd
SELECT_COLUMN = [1,2,4,6,11] # We only consider price, date of the purchase, property type, lease duration and town
df = pd.read_csv("pp-complete.csv", names=["price", "date", "ptype", "duration", "town"], sep=",", usecols=SELECT_COLUMN)

# Data Preprocessing
for c in ['duration', 'ptype']:
    df[c] = df[c].astype('category')
    df[c] = df[c].cat.codes

df['isLondon'] = [1 if i == "LONDON" else 0 for i in df['town']]
df['date'] =  pd.to_datetime(df['date'], format='%Y-%m-%d')
df["year"] = df['date'].dt.year

# Split the data into training/testing sets
df_train = df[df.date < "2016-1-1"]
df_test = df[df.date >= "2016-1-1"]
df_test = df_test[df.date < "2017-1-1"]
x_train = df_train[['year', 'ptype', 'duration', 'isLondon']].values
# x_train = df_train[['ptype']].values
y_train = df_train['price'].values

x_test = df_test[['year', 'ptype', 'duration', 'isLondon']].values
# x_test = df_test[['ptype']].values
y_test = df_test['price'].values

import keras
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# define the model
def baseline():
    # create model
    model = Sequential()
    model.add(Dense(4, input_dim=4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
model = baseline()
for i in range(100):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_train)
    print('Mean squared error: %.2f'
          % mean_squared_error(y_train, y_pred))

