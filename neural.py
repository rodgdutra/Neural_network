

# x is the input layer
# y_hat is the output layer
# W is the weights 
# b is the 


import numpy as np
import matplotlib.pyplot as plt 
import matplotlib

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],len(x)) 
        self.weights2   = np.random.rand(len(x),1)                 
        self.y          = y
        self.output     = np.zeros(self.y.shape)
        self.error = [0]

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))
        self.error_calc()

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def error_calc(self):
        error = 0
        for i in range(0,len(self.output)-1):
            error += self.y[i]-self.output[i]
        self.error.append(error)    

def transfer_function():
    x = np.arange(0,20,0.5)
    R = 1
    L = 5 
    y = R/L *np.exp(-x*R/L)  
    return x,y 

def main():
    matplotlib.style.use('classic') 
    x,y = transfer_function()
    
    x_factor = 1/np.max(x)
    if np.max(x) > 1 :
        x = np.array([x])*x_factor
    else:
        # Already normalized sample
        x = np.array([x]) 

    x = np.transpose(x)
                 
    y_factor = 1/np.max(y)
    if np.max(y) >  1 :
        y = np.array([y])*y_factor
    else:
        # Already normalized sample
        y = np.array([y]) 
    y = np.transpose(y)      
    nn = NeuralNetwork(x,y)

    for i in range(1500):
        nn.feedforward()
        nn.backprop()
    
    
    if np.max(y) > 1 :
        nn.output = nn.output*(1/y_factor)
        print("here")
    else:
        # Already normalized sample
        nn.output = nn.output

    plt.subplot(3, 1, 1)
    plt.plot(x,y, color='r', linewidth=1, linestyle='-', label='Real')
    plt.grid(color='k', linewidth=.5, linestyle=':')
    plt.legend(loc=4)
    plt.subplot(3, 1, 2)
    plt.plot(x,nn.output , 'b', linewidth=1, label='Neural')
    plt.grid(color='k', linewidth=.5, linestyle=':')
    plt.legend(loc=4)
    plt.subplot(3, 1, 3)
    n = np.arange(0,len(nn.error),1)
    plt.plot(n,nn.error , 'g', linewidth=1, label='error')
    plt.grid(color='k', linewidth=.5, linestyle=':')
    plt.legend(loc=4)
    plt.show()
if __name__ == "__main__":
    main()
   

