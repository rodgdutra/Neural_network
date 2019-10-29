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
# Generating the autocorrelation to each lag
x = np.array(wind_april).flatten()
x1 = np.array(wind_may).flatten()
lag = [0, 1, 2, 3, 4, 5, 17, 150]
start = lag[-1]
end = len(x)

# Lag matrix
R = [x[start - i : end - i] for i in lag]
r = np.corrcoef(R, R)
corr = r[0][:8]
print(corr)

plt.figure()
pd.plotting.autocorrelation_plot(x)
plt.show()

#%%
# Split the dataset
lag = [1, 2]
start = lag[-1]
end = 4320
x_train = [x[start - i : end - i] for i in lag]
y_train = x[start:end]

lag = [1, 2]
start = lag[-1]
end = 2200
x_val = [x1[start - i : end - i] for i in lag]
y_val = x1[start:end]

lag = [1, 2]
start = 2201
end = 4460
x_test = [x1[start - i : end - i] for i in lag]
y_test = x1[start:end]

#%%
# Normalize the datasets
min1 = np.min(x_train[0])
max1 = np.max(x_train[0])
min2 = np.min(x_train[1])
max2 = np.max(x_train[1])
x_train[0] = (x_train[0] - min1) / (max1 - min1)
x_train[1] = (x_train[1] - min2) / (max1 - min2)

min1 = np.min(x_val[0])
max1 = np.max(x_val[0])
min2 = np.min(x_val[1])
max2 = np.max(x_val[1])
x_val[0] = (x_val[0] - min1) / (max1 - min1)
x_val[1] = (x_val[1] - min2) / (max1 - min2)

min1 = np.min(x_test[0])
max1 = np.max(x_test[0])
min2 = np.min(x_test[1])
max2 = np.max(x_test[1])
x_test[0] = (x_test[0] - min1) / (max1 - min1)
x_test[1] = (x_test[1] - min2) / (max1 - min2)

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
plt.plot(y_test.flatten(), "r")
plt.plot(pred.flatten(), "b")
plt.show()
