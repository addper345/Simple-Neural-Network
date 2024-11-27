import numpy as np
from Neuron import sigmoid, Neuron

class NeuralNetwork:
    def __init__(self):
        self.h1 = Neuron()
        self.h2 = Neuron()
        self.o1 = Neuron()

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

        h1_deriv = foo_deriv*input[0]
        w1_deriv = loss_deriv*y_pred_deriv*h1_deriv

        h1_deriv = foo_deriv*input[1]
        w2_deriv = loss_deriv*y_pred_deriv*h1_deriv

        y_pred_deriv = fx_deriv*self.o1.weights[1]
        foo = self.h2.feedForward(input)
        foo_deriv = foo*(1-foo)

        h2_deriv = foo_deriv*input[0]
        w3_deriv = loss_deriv*y_pred_deriv*h2_deriv

        h2_deriv = foo_deriv*input[1]
        w4_deriv = loss_deriv*y_pred_deriv*h2_deriv

        return [w1_deriv, w2_deriv, w3_deriv, w4_deriv]

        
        


    
    
