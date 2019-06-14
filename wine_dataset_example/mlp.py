import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from pprint import pprint

class MLP():
	def __init__(self,h_n=50,data_csv='wine2.csv'):
		self.load_data(data_csv)
		self.norm_dataset(0,13)
		self.baseline_model(h_n)

	def norm_dataset(self,start,end):
		dataset = list()

		for i in range(start,end):
			norm_col = self.dataset[:,i]/np.amax(self.dataset[:,i])
			dataset.append(norm_col)

		for i in range(end,end+2):
			col = self.dataset[:,i]
			dataset.append(col)

		dataset = pd.DataFrame(dataset)
		dataset = dataset.transpose()
		self.dataset =dataset.values

	def load_data(self,data_csv):
		# load dataset
		dataframe = pd.read_csv(data_csv, header=None)
		self.dataset = dataframe.values
		self.X = self.dataset[:,0:13].astype(float)
		self.Y = self.dataset[:,13:15]

	def baseline_model(self,h_n):
		# create model
		self.model = Sequential()
		self.model.add(Dense(h_n, input_dim=13,
							 activation='sigmoid'))
		self.model.add(Dense(1, activation='linear'))

		# Compile model
		self.model.compile(loss='mse',
						   optimizer='ADAM',
						   metrics=['accuracy'])

	def split_data(self,seed=9,random=True):
		if (random):
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
			print(self.X_train.shape)

	def set_data(self,X,X_train,X_val,X_test,Y,Y_train,Y_val,Y_test):
		self.X       = X
		self.X_train = X_train
		self.X_val   = X_val
		self.X_test  = X_test
		self.Y       = Y
		self.Y_train = Y_train
		self.Y_val   = Y_val
		self.Y_test  = Y_test

	def train_and_val(self):
		# Train and validate the model
		self.model.fit(self.X_train,
					   self.Y_train[:,0],
					   validation_data=(self.X_val,self.Y_val[:,0]),
					   epochs=400, batch_size=10,verbose=0)
	def test(self):
		self.net_test = np.round([i[0] for i in self.model.predict(self.X_test)])

	def total_output(self):
		self.net_out = np.round([i[0] for i in self.model.predict(self.X)])

	def score(self):
		scores = self.model.evaluate(self.X_test, self.Y_test[:,0])
		# evaluate the model
		print("\n%s: %.2f%%" % (self.model.metrics_names[1], scores[1]*100))
		return scores[1]
