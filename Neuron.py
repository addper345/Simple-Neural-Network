import numpy as np

def sigmoid(x): 
    return 1/(1+np.exp(-x))

class Neuron:
    def __init__(self):
        self.weights = [0,0]
        self.weights[0] = np.random.normal()
        self.weights[1] = np.random.normal()
        self.bias = 0

    def feedForward(self, input):
        total = sigmoid(np.dot(self.weights, np.array(input, dtype="float")) + self.bias)
        return total
    

        
    


