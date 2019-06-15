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


class Auto_associative_mlp():
	def __init__(self,h_n=5,data_csv='wine2.csv'):
		self.load_data(data_csv)
		self.norm_dataset(0,13)
		self.X = self.dataset[:,0:13].astype(float)
		self.Y = self.dataset[:,13:15]
		self.X1,self.Y1 = self.split_class_dataset(0,59)
		self.X2,self.Y2 = self.split_class_dataset(59,130)
		self.X3,self.Y3 = self.split_class_dataset(130,178)

		self.model1 = self.baseline_model(h_n=h_n)
		self.model2 = self.baseline_model(h_n=h_n,dec_act='linear')
		self.model3 = self.baseline_model(h_n=h_n)

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

	def baseline_model(self,h_n=5,dec_act='sigmoid'):
		# create model
		model = Sequential()
		model.add(Dense(h_n, input_dim=13, activation=dec_act))
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

	def get_data(self):
		X_train = np.concatenate((self.X_train1,self.X_train2,self.X_train3))
		X_val   = np.concatenate((self.X_val1,self.X_val2,self.X_val3))
		X_test  = np.concatenate((self.X_test1,self.X_test2,self.X_test3))
		self.X_test = X_test
		Y_train = np.concatenate((self.Y_train1,self.Y_train2,self.Y_train3))
		Y_val   = np.concatenate((self.Y_val1,self.Y_val2,self.Y_val3))
		Y_test  = np.concatenate((self.Y_test1,self.Y_test2,self.Y_test3))

		return self.X,X_train,X_val,X_test,self.Y,Y_train,Y_val,Y_test

	def train_and_val(self,model,X,Y,X_val,Y_val):
		# Train and validate the model
		model.fit(X,Y,
				  validation_data=(X_val,Y_val),
				  batch_size=200,
				  epochs=2000,
				  shuffle=False,
				  verbose=0)

	def train_procedure(self,id_i):
		model  = getattr(self,'model'+id_i)
		train  = getattr(self,'X_train'+id_i)
		val    = getattr(self,'X_val'+id_i)

		self.train_and_val(model,train,
						   train,
						   val,
						   val)

	def score(self):
		score  = list()
		score.append(self.model1.evaluate(self.X_test1,self.X_test1)[0])
		score.append(self.model2.evaluate(self.X_test2,self.X_test2)[0])
		score.append(self.model3.evaluate(self.X_test3,self.X_test3)[0])

		return score

	def model_out(self):
		print("")

	def total_out(self,test=False):
		if test == False:
			Y = getattr(self,'Y')
			X = getattr(self,'X')
			index  = Y[:,1]

		if test == True:
			Y1 = getattr(self,'Y_test1')
			Y2 = getattr(self,'Y_test2')
			Y3 = getattr(self,'Y_test3')
			Y  = np.concatenate((Y1,Y2))
			Y  = np.concatenate((Y,Y3))
			X1 = getattr(self,'X_test1')
			X2 = getattr(self,'X_test2')
			X3 = getattr(self,'X_test3')
			X  = np.concatenate((X1,X2))
			X  = np.concatenate((X,X3))
			index  = np.append(Y1[:,1],Y2[:,1])
			index  = np.append(index,Y3[:,1])

		error_1 = list()
		error_2 = list()
		error_3 = list()

		error = list()
		out = list()

		for x in range(0,len(Y[:,0])):
			y = x+1
			scores = self.model1.evaluate(X[x:y],X[x:y])
			error_1.append(scores[0])

		for x in range(0,len(Y[:,0])):
			y = x+1
			scores = self.model2.evaluate(X[x:y],X[x:y])
			error_2.append(scores[0])

		for x in range(0,len(Y[:,0])):
			y = x+1
			scores = self.model3.evaluate(X[x:y],X[x:y])
			error_3.append(scores[0])

		error.append(error_1)
		error.append(error_2)
		error.append(error_3)
		error = pd.DataFrame(error)
		error = error.transpose()
		error = error.values

		if test == True:
			xt1 = np.arange(0,12)
			xt2 = np.arange(12,25)
			xt3 = np.arange(25,36)
			# Figure 1
			plt.figure()
			plt.plot(error_1,'r',marker='v',label='Erro RNA 1')
			x1 = np.arange(0,11)
			plt.plot(xt1,error_1[0:12],'b.',label='teste classe 1')
			plt.plot(xt2,error_1[12:25],'y.',label='teste classe 2')
			plt.plot(xt3,error_1[25:36],'c.',label='teste classe 3')
			plt.legend()
			plt.ylabel('Erro quadrático médio')
			plt.xlabel('Index de teste')
			plt.grid()
			plt.savefig('plots/Erro_classe1_auto.png')
			# Figure 2
			plt.figure()
			plt.plot(error_2,'r',marker='v',label='Erro RNA 2')
			plt.plot(xt1,error_2[0:12],'b.',label='teste classe 1')
			plt.plot(xt2,error_2[12:25],'y.',label='teste classe 2')
			plt.plot(xt3,error_2[25:36],'c.',label='teste classe 3')
			plt.legend()
			plt.ylabel('Erro quadrático médio')
			plt.xlabel('Index de teste')
			plt.grid()
			plt.savefig('plots/Erro_classe2_auto.png')
			# Figure 3
			plt.figure()
			plt.plot(error_3,'r',marker='v',label='Erro RNA 3')
			plt.plot(xt1,error_3[0:12],'b.',label='teste classe 1')
			plt.plot(xt2,error_3[12:25],'y.',label='teste classe 2')
			plt.plot(xt3,error_3[25:36],'c.',label='teste classe 3')
			plt.legend()
			plt.ylabel('Erro quadrático médio')
			plt.xlabel('Index de teste')
			plt.grid()
			plt.savefig('plots/Erro_classe3_auto.png')
			print(len(self.Y_test1[:,0]),len(self.Y_test2[:,0]),len(self.Y_test3[:,0]))

		for i in range(0,len(error[:,0])):
			out_i = np.argmin(error[i])
			out.append(out_i)


		out    = np.add(out,1)

		return index,out