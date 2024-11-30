import numpy as np
from Neuron import Neuron
from csvReader import getTrainingData, averageHeight, averageWeight

def sigmoid(x):
  # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
  # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)
  return fx * (1 - fx)

class NeuralNetwork:
    def __init__(self):
        self.h1 = Neuron()
        self.h2 = Neuron()
        self.o1 = Neuron()
        self.h1.bias = -161
        self.h2.bias = -133

    def feedForward(self, input):
        return self.o1.feedForward([self.h1.feedForward(input), self.h2.feedForward(input)])
    
    def MSE_loss(y_pred, y_true):
        return ((y_pred-y_true) ** 2).mean()
    
    def calculate_deriv(self, y_pred, y_true, input):
        fx = self.feedForward(input)
        fx_deriv = fx*(1-fx)
        loss_deriv = -2*(y_true - y_pred)

        y_pred_deriv = fx_deriv*self.o1.weights[0]
        foo = self.h1.feedForward(input)
        foo_deriv = foo*(1-foo)

        h1_deriv = foo_deriv*int(input[0])
        w1_deriv = loss_deriv*y_pred_deriv*h1_deriv

        o_weight_1 = loss_deriv*fx_deriv*foo

        h1_deriv = foo_deriv*int(input[1])
        w2_deriv = loss_deriv*y_pred_deriv*h1_deriv

        y_pred_deriv = fx_deriv*self.o1.weights[1]
        foo = self.h2.feedForward(input)
        foo_deriv = foo*(1-foo)

        h2_deriv = foo_deriv*int(input[0])
        w3_deriv = loss_deriv*y_pred_deriv*h2_deriv

        o_weight_2 = loss_deriv*fx_deriv*foo

        h2_deriv = foo_deriv*int(input[1])
        w4_deriv = loss_deriv*y_pred_deriv*h2_deriv


        return [w1_deriv, w2_deriv, w3_deriv, w4_deriv, o_weight_1, o_weight_2]
    
    def train(self, input, trainingSpeed, epochs):
        for epoch in range(epochs):
            for array in input:
                y_pred = self.feedForward(array[1:])
                deriv = self.calculate_deriv(y_pred, array[0], array[1:])

                self.h1.weights[0] = -trainingSpeed*deriv[0]+self.h1.weights[0]
                self.h1.weights[1] = -trainingSpeed*deriv[1]+self.h1.weights[1]
                self.h2.weights[0] = -trainingSpeed*deriv[2]+self.h2.weights[0]
                self.h2.weights[0] = -trainingSpeed*deriv[3]+self.h2.weights[1]
                self.o1.weights[0] = -trainingSpeed*deriv[4]+self.o1.weights[0]
                self.o1.weights[1] = -trainingSpeed*deriv[5]+self.o1.weights[1]

    

neuralNet = NeuralNetwork()
array = getTrainingData(5)
print(array)
data = np.array([
  [1, -2, -1], 
  [0, 25, 6], 
  [0, 17, 4], 
  [1, -15, -6]
])
neuralNet.train(data, 0.1, 1000)
cont = 1
while cont==1:
    height = int(input("INPUT HEIGHT: "))
    weight = int(input("\nINPUT WEIGHT: "))
    inp = [height, weight]
    print(neuralNet.feedForward(inp))
    cont = int(input("\nDO YOU WANT TO CONTINUE? 1/0: "))









        

        
        


    
