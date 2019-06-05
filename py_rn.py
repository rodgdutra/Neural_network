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


class Back_prop_mlp():
	def __init__(self,data_csv='dados2.csv'):
		self.load_data(data_csv)
		self.baseline_model()

	def load_data(self,data_csv):
		# load dataset
		dataframe = pd.read_csv(data_csv, header=None)
		dataset = dataframe.values
		self.X = dataset[:,0:4].astype(float)
		self.Y = dataset[:,4:6]
		print(self.Y[:,0])

	def baseline_model(self,h_layers=15):
		# create model
		self.model = Sequential()
		self.model.add(Dense(h_layers, input_dim=4, activation='relu'))
		self.model.add(Dense(1, activation='relu'))

		# Compile model
		self.model.compile(loss='mse',
						   optimizer='adam',
						   metrics=['accuracy'])
	def split_data(self,seed=9):
		# Split data set into train and validation
		self.X_train, self.X_val,self.Y_train, self.Y_val = train_test_split(self.X,
																			  self.Y,
																			  test_size=0.2,
																			  random_state=seed)
		# Split train into train and test
		self.X_train, self.X_test,self.Y_train, self.Y_test = train_test_split(self.X_train,
																			   self.Y_train,
																			   test_size=0.25,
																			   random_state=seed)
	def train_and_val(self):
		# Train and validate the model
		self.model.fit(self.X_train,
					   self.Y_train[:,0],
					   validation_data=(self.X_val,self.Y_val[:,0]),
					   epochs=120, batch_size=10)
	def test(self):
		self.net_test = np.round([i[0] for i in self.model.predict(self.X_test)])

	def total_output(self):
		self.net_out = np.round([i[0] for i in self.model.predict(self.X)])

	def score(self):
		scores = self.model.evaluate(self.X_test, self.Y_test[:,0])
		# evaluate the model
		print("\n%s: %.2f%%" % (self.model.metrics_names[1], scores[1]*100))

def main():
	mlp = Back_prop_mlp()
	mlp.split_data()
	mlp.train_and_val()
	mlp.test()
	mlp.total_output()
	mlp.score()

	plt.figure()
	plt.plot(mlp.Y_test[:,1],mlp.net_test,'yd',label='net test')
	plt.plot(mlp.Y[:,1],mlp.net_out,'r*',label='net_output')
	plt.plot(mlp.Y[:,1],mlp.Y[:,0],'b.',label='true output')
	plt.grid()
	plt.legend()
	plt.show()


if __name__ == '__main__':
	main()
