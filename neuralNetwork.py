import numpy as np
from Neuron import sigmoid, Neuron

class NeuralNetwork:
    def __init__(self):
        self.h1 = Neuron()
        self.h2 = Neuron()
        self.o1 = Neuron()

    def feedForwards(self, input):
        return self.o1.feedForward([self.h1.feedForward(input), self.h2.feedForward(input)])
