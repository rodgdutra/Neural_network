#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

#%%
# Função de ativação
def sigmoid(t):
    return 1/(1+np.exp(-t))

# Derivada da função de ativação
def sigmoid_derivative(p):
    return p * (1 - p)

# Função para o calculo do erro médio quadratico de 2 arrays
def mse(y1,y2):
    return np.mean(np.square(y1 - y2))

def Vc_RC(t,r=5,c=0.1,vin=1):
    """
    Tensão de um capacitor em um circuito RC
    """
    tau = -t/(r*c)
    vc  = vin*(1 - np.exp(tau))

    return vc
#%%
# Class definition
class NeuralNetwork:
    def __init__(self, x, y, n=4):
        """
        Definição de um objeto de rede neural

        argumentos:
        x: a entrada de treino
        y: a saída desejada no treino
        n: Número de neurônios na camada escondida
        """
        self.entrada = x
        self.pesos1 = np.random.rand(self.entrada.shape[1],n)
        self.pesos2 = np.random.rand(4,1)
        self.y = y
        self.saida = np. zeros(y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.entrada, self.pesos1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.pesos2))
        # Nota-se que a saída da camada 2 é a saída do modelo
        return self.layer2

    def backprop(self):
        d_pesos2 = np.dot(self.layer1.T, 2*(self.y -self.saida)*sigmoid_derivative(self.saida))
        d_pesos1 = np.dot(self.entrada.T, np.dot(2*(self.y -self.saida)*sigmoid_derivative(self.saida), self.pesos2.T)*sigmoid_derivative(self.layer1))

        self.pesos1 += d_pesos1
        self.pesos2 += d_pesos2

    def train(self):
        self.saida = self.feedforward()
        self.backprop()
#%%
def main():
    # Definindo um modelo matemático
    t = np.arange(0,3,0.1)
    t = t.reshape(len(t),1)

    vc = Vc_RC(t)
    vc = vc.reshape(len(vc),1)

    # Definindo o objeto da rede neural
    nn_vc_model = NeuralNetwork(t,vc)


    # Treinando a rede por 500 epocas
    erro = list()

    for i in range(500):
        saida_rede = nn_vc_model.feedforward() # calculando a saida da rede
        erro.append( mse(vc, saida_rede)) # calculabdo o mse e guardando em um vetor
        nn_vc_model.train() # utilizando um metodo do objeto rede neural para treinar

    fig = plt.figure()
    plt.plot(erro,'r')
    plt.xlabel("epoca")
    plt.ylabel("erro quadratico médio")
    plt.show()

    # transformando as matrizes de entrada em vetores para a plotagem
    t          = t.flatten()
    vc         = vc.flatten()

    # Transformando a saida da rede neural para a plotagem
    saida_rede = saida_rede.flatten()

    fig = plt.figure()
    plt.plot(t, vc, 'b', label="tensão VC calculada")
    plt.plot(t, saida_rede, 'r', label="tensão VC rede")
    plt.legend()
    plt.grid()
    plt.xlabel("tempo")
    plt.ylabel("tensão")
    plt.show()

if __name__ == "__main__":
    main()