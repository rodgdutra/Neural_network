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
import matplotlib
#matplotlib.style.use('classic')
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

	def baseline_model(self,h_layers=50):
		# create model
		self.model = Sequential()
		self.model.add(Dense(h_layers, input_dim=13,
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

	def train_and_val(self):
		# Train and validate the model
		self.model.fit(self.X_train,
					   self.Y_train[:,0],
					   validation_data=(self.X_val,self.Y_val[:,0]),
					   epochs=400, batch_size=10,verbose=1)
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
		self.norm_dataset(0,13)
		self.X = self.dataset[:,0:13].astype(float)
		self.Y = self.dataset[:,13:15]
		self.X1,self.Y1 = self.split_class_dataset(0,59)
		self.X2,self.Y2 = self.split_class_dataset(59,130)
		self.X3,self.Y3 = self.split_class_dataset(130,178)

		self.model1 = self.baseline_model()
		self.model2 = self.baseline_model()
		self.model3 = self.baseline_model()

		self.split_dataset('1')
		self.split_dataset('2')
		self.split_dataset('3')


	def load_data(self,data_csv):
		# load dataset
		dataframe = pd.read_csv(data_csv, header=None)
		self.dataset = dataframe.values
		#pprint(dataframe)

	def split_class_dataset(self,start,end):
		X_i = self.dataset[start:end][:,0:13].astype(float)
		Y_i = self.dataset[start:end][:,13:15]
		return X_i,Y_i

	def baseline_model(self,h_layers=2):
		# create model
		model = Sequential()
		model.add(Dense(h_layers, input_dim=13, activation='linear'))
		model.add(Dense(13, activation='linear'))
		rms_prop = keras.optimizers.RMSprop(lr=1, rho=0.9, epsilon=None, decay=0)
		adam = keras.optimizers.Adam(lr=0.045, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
		# Compile model
		model.compile(loss='mean_squared_error', optimizer=adam,metrics=['acc'])
		return model

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

	def split_dataset(self,id_i,seed=9):

		X = getattr(self,'X'+id_i)
		Y = getattr(self,'Y'+id_i)

		train_x, val_x ,train_y, val_y  = train_test_split(X,Y,
															test_size=0.2,
															random_state=seed)
		# Split train into train and test
		train_x, test_x, train_y, test_y = train_test_split(train_x,train_y,
															test_size=0.25,
															random_state=seed)

		setattr(self,'X_train'+id_i,train_x)
		setattr(self,'X_test'+id_i,test_x )
		setattr(self,'X_val'+id_i,val_x)
		setattr(self,'Y_train'+id_i,train_y )
		setattr(self,'Y_test'+id_i,test_y )
		setattr(self,'Y_val'+id_i,val_y )

	def train_and_val(self,model,X,Y,X_val,Y_val):
		# Train and validate the model
		model.fit(X,Y,
				  validation_data=(X_val,Y_val),
				  batch_size=200,
				  epochs=3000,
				  shuffle=False)

	def train_procedure(self,id_i):
		model  = getattr(self,'model'+id_i)
		train  = getattr(self,'X_train'+id_i)
		val    = getattr(self,'X_val'+id_i)

		self.train_and_val(model,train,
						   train,
						   val,
						   val)

	def score(self,model,id,X_test,Y_test):
		scores = model.evaluate(X_test,Y_test)
		# evaluate the model
		print("error %f" %scores[0])

	def model_out(self):
		print("")

	def total_out(self):
		error_1 = list()
		error_2 = list()
		error_3 = list()

		error = list()
		self.out = list()

		for x in range(0,len(self.Y[:,0])):
			y = x+1
			scores = self.model1.evaluate(self.X[x:y],self.X[x:y])
			error_1.append(scores[0])

		for x in range(0,len(self.Y[:,0])):
			y = x+1
			scores = self.model2.evaluate(self.X[x:y],self.X[x:y])
			error_2.append(scores[0])

		for x in range(0,len(self.Y[:,0])):
			y = x+1
			scores = self.model3.evaluate(self.X[x:y],self.X[x:y])
			error_3.append(scores[0])

		error.append(error_1)
		error.append(error_2)
		error.append(error_3)
		error = pd.DataFrame(error)
		error = error.transpose()
		error = error.values
		plt.figure()
		plt.plot(error_2)
		plt.show()
		for i in range(0,len(error[:,0])):
			out_i = np.argmin(error[i])
			self.out.append(out_i)

		self.out = np.add(self.out,1)
		#print(len(self.Y_test1[:,0]),len(self.Y_test2[:,0]),len(self.Y_test3[:,0]))

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

def auto_mlp_test():

	auto_mlp = Auto_associative_mlp()
	auto_mlp.train_procedure('1')
	auto_mlp.train_procedure('2')
	auto_mlp.train_procedure('3')
	auto_mlp.total_out()
	plt.plot(auto_mlp.Y[:,1],auto_mlp.out,'r*',label='net_output')
	plt.plot(auto_mlp.Y[:,1],auto_mlp.Y[:,0],'b.',label='true output')
	plt.grid()
	plt.legend()

def main():
	backprop_test()
	#auto_mlp_test()
	plt.show()
if __name__ == '__main__':
	main()
