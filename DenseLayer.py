from numpy import dot, zeros, sum
from numpy.random import randn


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, activation_function):
        self.activation_function = activation_function
        self.weights = 0.10 * randn(n_inputs, n_neurons)
        self.biases = zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        x = dot(inputs, self.weights) + self.biases
        self.output = self.activation_function().forward(x)
        return self.output
    
    def backward(self, dvalues):
        self.dweights = dot(self.inputs.T, dvalues)
        self.dbiases = sum(dvalues, axis=0, keepdims=True)
        self.dinputs = dot(dvalues, self.weights.T)