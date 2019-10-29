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
data_csv = "mamografy.csv"
dataframe = pd.read_csv(data_csv, header=None)
dataset = dataframe.values

# Normalize the dataset
start = 0
end = 5
norm_dataset = list()

for i in range(start,end):
    norm_col = dataset[:,i]/np.amax(dataset[:,i])
    norm_dataset.append(norm_col)

for i in range(end,end+2):
    col = dataset[:,i]
    norm_dataset.append(col)

norm_dataset = pd.DataFrame(norm_dataset)
norm_dataset = norm_dataset.transpose()
norm_dataset = norm_dataset.values

dataset = norm_dataset

x = dataset[:, 0:5].astype(float)
y = dataset[:, 5:7]

#%%
# Spliting data

seed = 42
# Split data set into train and validation
x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.15, random_state=seed
)
# Split train into train and test
x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, test_size=0.175, random_state=seed
)

#%%
# Building the NN
input_dim = x.shape[1]
h_n = 10
model = Sequential()
model.add(Dense(h_n, input_dim=input_dim, activation="sigmoid"))
model.add(Dense(1, activation="linear"))

# Compile model
model.compile(loss="mse", optimizer="ADAM", metrics=["accuracy"])

# Train and validate the model
history = model.fit(
    x_train,
    y_train[:, 0],
    validation_data=(x_val, y_val[:, 0]),
    epochs=200,
    batch_size=30,
    # verbose=0,
    callbacks=[
        EarlyStopping(
            monitor="val_loss", mode="min", min_delta=1, patience=10, verbose=1
        )
    ],
)

# Plot the validation and train loss
plt.plot(np.sqrt(history.history['loss']))
plt.plot(np.sqrt(history.history['val_loss']))
plt.title('Model loss')
plt.ylabel('sqrt(MSE)')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

pred = model.predict(x_val)
pred = np.array(pred).flatten()

erro = pred - np.array(y_val[:,0]).flatten()
erro = np.abs(erro)

pprint(erro)

acerto = 0
for i in erro:
    if i < 0.5:
        acerto +=1

print(acerto)
print(len(erro))