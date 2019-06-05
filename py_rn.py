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
	def __init__(self,data_csv='wine2.csv'):
		self.load_data(data_csv)
		self.baseline_model()

	def load_data(self,data_csv):
		# load dataset
		dataframe = pd.read_csv(data_csv, header=None)
		self.dataset = dataframe.values
		self.X = self.dataset[:,0:13].astype(float)
		self.Y = self.dataset[:,13:15]

	def baseline_model(self,h_layers=60):
		# create model
		self.model = Sequential()
		self.model.add(Dense(h_layers, input_dim=13,
							 activation='sigmoid'))
		self.model.add(Dense(1, activation='relu'))

		# Compile model
		self.model.compile(loss='mse',
						   optimizer='adam',
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

class Auto_associative_mlp():
	def __init__(self,data_csv='wine2.csv'):
		self.load_data(data_csv)
		self.X1,self.Y1 = self.split_class_dataset(0,60)
		self.X2,self.Y2 = self.split_class_dataset(60,131)
		self.X3,self.Y3 = self.split_class_dataset(131,178)

		self.model1 = self.baseline_model()
		self.model2 = self.baseline_model()
		self.model3 = self.baseline_model()

		self.X_train1, self.X_val1,self.Y_train1, self.Y_val1   = self.split_dataset(self.X1,
																					   self.Y1)
		self.X_train1, self.X_test1,self.Y_train1, self.Y_test1 = self.split_dataset(self.X_train1,
																						self.Y_train1)
		self.X_train2, self.X_val2,self.Y_train2, self.Y_val2   = self.split_dataset(self.X2,
																					   self.Y2)
		self.X_train2, self.X_test2,self.Y_train2, self.Y_test2 = self.split_dataset(self.X_train2,
																						self.Y_train2)
		self.X_train3, self.X_val3,self.Y_train3, self.Y_val3   = self.split_dataset(self.X3,
																					   self.Y3)
		self.X_train3, self.X_test3,self.Y_train3, self.Y_test3 = self.split_dataset(self.X_train3,
																						self.Y_train3)
		print(self.X_val1.shape)
		self.train_and_val(self.model1,self.X_train1,
						   self.X_train1,
						   self.X_val1,
						   self.X_val1)
		self.score(self.model1,'1',self.X_test1, self.Y_test1)

	def load_data(self,data_csv):
		# load dataset
		dataframe = pd.read_csv(data_csv, header=None)
		self.dataset = dataframe.values

	def split_class_dataset(self,start,end):
		X_i = self.dataset[start:end][:,0:13].astype(float)
		Y_i = self.dataset[start:end][:,13:15]
		return X_i,Y_i

	def baseline_model(self,h_layers=50):
		# create model
		model = Sequential()
		model.add(Dense(h_layers, input_dim=13, activation='relu'))
		model.add(Dense(13, activation='sigmoid'))

		# Compile model
		model.compile(loss='mse',
						   optimizer='adam',
						   metrics=['accuracy'])
		return model

	def split_dataset(self,X,Y,seed=9):
		# Split data set into train and validation
		X_train, X_val, Y_train, Y_val = train_test_split(X,Y,
														  test_size=0.2,
														  random_state=seed)
		return X_train, X_val, Y_train, Y_val
		# Split train into train and test
		self.X_train, self.X_test,self.Y_train, self.Y_test = train_test_split(self.X_train,
																			   self.Y_train,
																			   test_size=0.25,
																			   random_state=seed)
	def train_and_val(self,model,X,Y,X_val,Y_val):
		# Train and validate the model
		model.fit(X,Y[:,0],
				  validation_data=(X_val,Y_val[:,0]),
				  epochs=120, batch_size=10)

	def score(self,model,id,X_test,Y_test):
		scores = model.evaluate(X_test,Y_test[:,0])
		# evaluate the model
		print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

def backprop_test():
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

def main():
	auto_mlp = Auto_associative_mlp()


if __name__ == '__main__':
	main()
