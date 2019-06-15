import matplotlib.pyplot as plt
import matplotlib
from mlp import MLP
import numpy as np
import pandas as pd
from autoencoder_mlp import Auto_associative_mlp
from pprint import pprint
import time
#matplotlib.style.use('classic')

def backprop_test():
	mlp = MLP()
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
	id_out, out = auto_mlp.total_out()
	id_test, test = auto_mlp.total_out(test=True)
	plt.plot(id_test,test,'yd',label='net test')
	plt.plot(id_out,out,'r*',label='net_output')
	plt.plot(auto_mlp.Y[:,1],auto_mlp.Y[:,0],'b.',label='true output')
	plt.grid()
	plt.legend()

def train_score_routine():
	h_n = [240]
	rep = 10
	error = list()
	h_ni  = list()
	auto_mlp = Auto_associative_mlp(h_n=5)
	auto_mlp.train_procedure('1')
	auto_mlp.train_procedure('2')
	auto_mlp.train_procedure('3')
	X,X_train,X_val,X_test,Y,Y_train,Y_val,Y_test = auto_mlp.get_data()
	for i in h_n:
		ej = 0
		acc_100 =0
		for j in range(0,rep):
			mlp = MLP(i)
			mlp.set_data(X,X_train,X_val,X_test,Y,Y_train,Y_val,Y_test)
			mlp.train_and_val()
			mlp.test()
			ej = np.subtract(mlp.Y_test[:,0],mlp.net_test)
			total = len(Y_test[:,0])
			acc = len(np.where(ej == 0)[0])
			acc_100 += acc*100.0/total

		error.append(acc_100/rep)
		h_ni.append(i)
	table = list()
	table.append(error)
	table.append(h_ni)
	table = pd.DataFrame(table)
	table.append(error)
	table = table.transpose()
	table = table.values
	print("Tabela")
	pprint(table)

def train_score_automlp():
	h_n = [4]
	error = list()
	h_ni  = list()
	rep = 10
	for i in h_n:
		acc_100 = 0
		for j in range(0,rep):
			auto_mlp = Auto_associative_mlp(h_n=i)
			auto_mlp.train_procedure('1')
			auto_mlp.train_procedure('2')
			auto_mlp.train_procedure('3')
			X,X_train,X_val,X_test,Y,Y_train,Y_val,Y_test = auto_mlp.get_data()
			id_test, test = auto_mlp.total_out(test=True)
			ej = np.subtract(Y_test[:,0],test)
			total = len(Y_test[:,0])
			acc = len(np.where(ej == 0)[0])
			acc_100 += acc*100.0/total

		error.append(acc_100/rep)
		h_ni.append(i)
	table = list()
	table.append(error)
	table.append(h_ni)
	table = pd.DataFrame(table)
	table.append(error)
	table = table.transpose()
	table = table.values
	print("Tabela")
	pprint(table)
def compare_nets():
	start = time.time()
	auto_mlp = Auto_associative_mlp(h_n=4)
	auto_mlp.train_procedure('1')
	auto_mlp.train_procedure('2')
	auto_mlp.train_procedure('3')
	id_out, out = auto_mlp.total_out()
	id_test, test = auto_mlp.total_out(test=True)
	X,X_train,X_val,X_test,Y,Y_train,Y_val,Y_test = auto_mlp.get_data()
	tempo1 = time.time()- start
	start = time.time()
	mlp = MLP(h_n=240)
	mlp.set_data(X,X_train,X_val,X_test,Y,Y_train,Y_val,Y_test)
	mlp.train_and_val()
	mlp.test()
	mlp.total_output()
	mlp.score()
	tempo2 = time.time()- start
	print(tempo1)
	print(tempo2)
	plt.figure()
	plt.plot(mlp.Y_test[:,1],mlp.net_test,'r^',label='RNA MLP')
	plt.plot(mlp.Y_test[:,1],mlp.Y_test[:,0],'b.',label='Classe real correspondente')
	plt.xlabel('Index de entrada da rede')
	plt.ylabel('Classe da saída')
	plt.grid()
	plt.legend()
	plt.savefig('plots/MLP.png')

	plt.figure()
	plt.plot(id_test,test,'r^',label='RNA autoassociativa')
	plt.plot(mlp.Y_test[:,1],mlp.Y_test[:,0],'b.',label='Classe real correspondente')
	plt.xlabel('Index de entrada da rede')
	plt.ylabel('Classe da saída')
	plt.grid()
	plt.legend()
	plt.savefig('plots/auto_mlp_out.png')

	plt.figure()
	plt.plot(mlp.Y_test[:,1],mlp.net_test,'g^',label='RN mlp')
	plt.plot(id_test,test,'rv',label='RN autoassociativa')
	plt.plot(mlp.Y[:,1],mlp.Y[:,0],'b.',label='true output')
	plt.xlabel('Index de entrada da rede')
	plt.ylabel('Classe da saída')
	plt.grid()
	plt.legend()
	plt.savefig('plots/saida_comparativa.png')

def main():
	#backprop_test()
	#auto_mlp_test()
	#plt.show()
	compare_nets()
	#train_score_automlp()
	#train_score_routine()
if __name__ == '__main__':
	main()

"""
array([[  0.59126984,   1.        ],
       [  0.91269841,   5.        ],
       [  0.92460317,  10.        ],
       [  0.92857143,  15.        ],
       [  0.93650794,  30.        ],
       [  0.94444444,  60.        ],
       [  0.9484127 , 240.        ],
       [  0.94047619, 480.        ]])
"""
"""
Tabela
array([[ 94.44444444,   1.        ],
       [100.        ,   4.        ],
       [ 91.11111111,   8.        ],
       [ 69.16666667,  12.        ]])
"""
