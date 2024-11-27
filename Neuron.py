import numpy as np

def sigmoid(x): 
    return 1/(1+np.exp(-x))

class Neuron:

    def __init__ (self, weights=[1,0], bias=0):
        self.weights = weights
        self.bias = bias

    def feedForward (self, input):
        return sigmoid(np.dot(self.weights, input) + self.bias)
    


