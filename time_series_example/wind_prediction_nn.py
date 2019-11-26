#%%
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from pprint import pprint
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

#%%
# Importing data
data_csv = "wind_1.csv"
dataframe = pd.read_csv(data_csv, header=None)
wind_april = dataframe.values

data_csv = "wind_2.csv"
dataframe = pd.read_csv(data_csv, header=None)
wind_may = dataframe.values

#%%
# Check the autocorrelation
# Get the linear correlation coefficient to check which lags to use
# Generating the autocorrelation to each Lags
wind_april = np.array(wind_april).flatten()
wind_may   = np.array(wind_may).flatten()
Lags = [0, 1, 2, 3, 4, 5, 17, 150]
start = Lags[-1]
end = len(wind_april)

# Lag matrix
lag_matrix = [wind_april[start - i : end - i] for i in Lags]
linear_corr = np.corrcoef(lag_matrix, lag_matrix)[0][:len(Lags)]
print(" Lags analysed      :{0}".format(Lags))
print(" Linear correlation :{0}".format(linear_corr))

plt.figure()
pd.plotting.autocorrelation_plot(wind_april)
plt.show()

#%%
# Split the dataset in train, validation and test
# Split the dataset
Lags = [1, 2]
start = Lags[-1]
end = 4320
x_train = [wind_april[start - i : end - i] for i in Lags]
y_train = wind_april[start:end]

Lags = [1, 2]
start = Lags[-1]
end = 2200
x_val = [wind_may[start - i : end - i] for i in Lags]
y_val = wind_may[start:end]

Lags = [1, 2]
start = 2201
end = 4460
x_test = [wind_may[start - i : end - i] for i in Lags]
y_test = wind_may[start:end]

#%%
# Normalize the datasets
min1 = np.min(x_train[0])
mawind_may = np.max(x_train[0])
min2 = np.min(x_train[1])
max2 = np.max(x_train[1])
x_train[0] = (x_train[0] - min1) / (mawind_may - min1)
x_train[1] = (x_train[1] - min2) / (mawind_may - min2)

min1 = np.min(x_val[0])
mawind_may = np.max(x_val[0])
min2 = np.min(x_val[1])
max2 = np.max(x_val[1])
x_val[0] = (x_val[0] - min1) / (mawind_may - min1)
x_val[1] = (x_val[1] - min2) / (mawind_may - min2)

min1 = np.min(x_test[0])
mawind_may = np.max(x_test[0])
min2 = np.min(x_test[1])
max2 = np.max(x_test[1])
x_test[0] = (x_test[0] - min1) / (mawind_may - min1)
x_test[1] = (x_test[1] - min2) / (mawind_may - min2)

x_test = np.transpose(x_test)
x_train = np.transpose(x_train)
x_val = np.transpose(x_val)

y_test = np.transpose(y_test)
y_train = np.transpose(y_train)
y_val = np.transpose(y_val)

#%%
# Building the NN
input_dim = 2
h_n = 5
model = Sequential()
model.add(Dense(h_n, input_dim=input_dim, activation="sigmoid"))
model.add(Dense(1, activation="linear"))

# Compile model
model.compile(loss="mse", optimizer="ADAM", metrics=["accuracy"])

# Train and validate the model
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=200,
    batch_size=30,
    # verbose=0,
    callbacks=[
        EarlyStopping(
            monitor="val_loss", mode="min", min_delta=1, patience=10, verbose=1
        )
    ],
)

#%%
# Plot the validation and train loss
plt.plot(np.sqrt(history.history["loss"]))
plt.plot(np.sqrt(history.history["val_loss"]))
plt.title("Model loss")
plt.ylabel("sqrt(MSE)")
plt.xlabel("Epoch")
plt.legend(["Train", "Test"], loc="upper left")
plt.show()

pred = model.predict(x_val)
pred = np.array(pred).flatten()

plt.figure()
plt.plot(y_test.flatten(), "linear_corr")
plt.plot(pred.flatten(), "b")
plt.show()
