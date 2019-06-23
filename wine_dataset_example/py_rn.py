import matplotlib.pyplot as plt
import matplotlib
from mlp import MLP
import numpy as np
import pandas as pd
from autoencoder_mlp import Auto_associative_mlp
from anfis_net import Anfis_net
from pprint import pprint
import time
#matplotlib.style.use('classic')

def standard_mlp():
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

def train_score_auto_mlp():
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

def train_score_anfis():
	h_n = [1,5,10,15,30,60,120,240,480]
	rep = 10
	error = list()
	h_ni  = list()
	auto_mlp = Auto_associative_mlp(h_n=5)
	X,X_train,X_val,X_test,Y,Y_train,Y_val,Y_test = auto_mlp.get_data()
	for i in h_n:
		ej = 0
		acc_100 =0
		for j in range(0,rep):
			anfis_wine = None
			anfis_wine = Anfis_net(set_data=True,norm=False)
			anfis_wine.set_data(X,X_train,X_val,X_test,Y,Y_train,Y_val,Y_test)
			anfis_wine.baseline_model(i)
			anfis_wine.train_and_val()

			ej = np.subtract(anfis_wine.Y_test[:,0],
							 np.transpose(anfis_wine.pred))
			total = len(Y_test[:,0])
			acc = len(np.where(ej == 0)[0])
			print(i)
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
	anfis_wine = Anfis_net(set_data=True,norm=False)
	anfis_wine.set_data(X,X_train,X_val,X_test,Y,Y_train,Y_val,Y_test)
	anfis_wine.baseline_model(240)
	anfis_wine.train_and_val()

	print(tempo1)
	print(tempo2)
	plt.figure()
	plt.subplot(2, 1, 1)
	plt.plot(anfis_wine.Y_test[:,1],anfis_wine.pred,'r^',label='ANFIS')
	plt.plot(anfis_wine.Y_test[:,1],anfis_wine.Y_test[:,0],'b.',label='Classe real correspondente')
	plt.xlabel('Index de entrada da rede')
	plt.ylabel('Classe da saída')
	plt.grid()
	plt.legend()
	plt.subplot(2, 1, 2)
	plt.plot(np.squeeze(anfis_wine.trn_costs),'b',label="Treino")
	plt.plot(np.squeeze(anfis_wine.val_costs),'r',label="Validação")
	plt.xlabel('Épocas')
	plt.ylabel('Erro quadrático médio')
	plt.yscale('log')
	plt.grid()
	plt.legend()
	plt.show()
	plt.savefig('plots/Anfis.png')

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
	compare_nets()

if __name__ == '__main__':
	main()
