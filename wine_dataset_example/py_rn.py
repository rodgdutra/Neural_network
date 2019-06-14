import matplotlib.pyplot as plt
import matplotlib
from mlp import MLP
import numpy as np
import pandas as pd
from autoencoder_mlp import Auto_associative_mlp
from pprint import pprint
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

def train_score_routine(mlp):
	h_n = [1,5,10,15,30,60,120,240,480]
	rep = 7
	error = list()
	h_ni  = list()

	for i in h_n:
		ej = 0
		for j in range(0,rep):
			mlp = MLP(i)
			mlp.set_data(X,X_train,X_val,X_test,Y,Y_train,Y_val,Y_test)
			mlp.train_and_val()
			mlp.test()
			mlp.total_output()
			print(i)
			ej += mlp.score()

		ej = ej/rep
		error.append(ej)
		h_ni.append(i)
		table = list()
		table.append(error)
		table.append(h_ni)
		table = pd.DataFrame(table)
		table = table.transpose()
		table = table.values
		print("Tabela")
		pprint(table)

def compare_nets():

	auto_mlp = Auto_associative_mlp()
	auto_mlp.train_procedure('1')
	auto_mlp.train_procedure('2')
	auto_mlp.train_procedure('3')
	id_out, out = auto_mlp.total_out()
	id_test, test = auto_mlp.total_out(test=True)
	X,X_train,X_val,X_test,Y,Y_train,Y_val,Y_test = auto_mlp.get_data()

	mlp = MLP(h_n=240)
	mlp.set_data(X,X_train,X_val,X_test,Y,Y_train,Y_val,Y_test)
	mlp.train_and_val()
	mlp.test()
	mlp.total_output()
	mlp.score()


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
if __name__ == '__main__':
	main()

"""
array([[  0.59126984,   1.        ],
       [  0.91269841,   5.        ],
       [  0.92460317,  10.        ],
       [  0.92857143,  15.        ],
       [  0.93650794,  30.        ],
       [  0.94444444,  60.        ],
       [  0.94444444, 120.        ],
       [  0.9484127 , 240.        ],
       [  0.94047619, 480.        ]])
"""