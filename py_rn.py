import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from pprint import pprint
import matplotlib.pyplot as plt

#import matplotlib.pyplot as plt

# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(8, input_dim=4, activation='relu'))
	model.add(Dense(1, activation='relu'))

	# Compile model
	model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['accuracy'])
	return model


# load dataset
dataframe = pd.read_csv("dados.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]
id_v = np.arange(0,len(Y))

model = baseline_model()
model.fit(X, Y, epochs=150, batch_size=10)
# evaluate the model
scores = model.evaluate(X, Y)

predictions = np.round([i[0] for i in model.predict(X)])
print(predictions)
plt.figure()
plt.plot(id_v,predictions,'r*',label='net_output')
plt.plot(id_v,Y,'b.',label='true output')
plt.legend()
plt.show()
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
