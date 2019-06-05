import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from pprint import pprint
import matplotlib.pyplot as plt


# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(15, input_dim=4, activation='relu'))
	model.add(Dense(1, activation='relu'))

	# Compile model
	model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['accuracy'])
	return model

def main():
	# load dataset
	dataframe = pd.read_csv("dados2.csv", header=None)
	dataset = dataframe.values
	X = dataset[:,0:4].astype(float)
	Y = dataset[:,4:6]
	id_v = np.arange(0,len(Y))
	model = baseline_model()
	print(Y[:,0])



	# split the dataset
	seed = 20
	X_train, X_val, y_train, y_val = train_test_split(X, Y,
													  test_size=0.2,
													  random_state=seed)

	X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,
														test_size=0.25,
														random_state=seed)
	# Train and validate
	model.fit(X_train, y_train[:,0],validation_data=(X_val,y_val[:,0]), epochs=150, batch_size=10)

	# Test
	net_test = np.round([i[0] for i in model.predict(X_test)])

	# Net out for all dataset
	net_out = np.round([i[0] for i in model.predict(X)])

	scores = model.evaluate(X_test, y_test[:,0])

	# evaluate the model
	print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	print(y_test[:,1])
	plt.figure()
	plt.plot(y_test[:,1],net_test,'yd',label='net test')
	plt.plot(id_v,net_out,'r*',label='net_output')
	plt.plot(id_v,Y[:,0],'b.',label='true output')
	plt.grid()
	plt.legend()
	plt.show()


if __name__ == '__main__':
	main()

