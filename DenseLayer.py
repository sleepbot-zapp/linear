from numpy import dot, zeros, sum, argmax
from numpy.random import randn
from ActivationFunction import Activation_Softmax
from LossFunctions import Loss_CategoricalCrossEntropy

# drelu_dx = drelu_dsum * dsum_dmul * dmul_dx

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, activation_function):
        self.activation_function = activation_function()
        self.weights = 0.10 * randn(n_inputs, n_neurons)
        self.biases = zeros((1, n_neurons))

    def forward(self, inputs, *, y_true=None):
        self.inputs = inputs
        x = dot(inputs, self.weights) + self.biases
        if not isinstance(self.activation_function, Activation_Softmax_Loss_CategoricalCrossentropy):
            self.output = self.activation_function.forward(x)
        else:
            self.output = self.activation_function.forward(x, y_true)
        return self.output
    
    def activation_function_backward(self, dvalues):
        self.dinputs = self.activation_function.backward(dvalues)

    def backward(self, dvalues):
        self.dweights = dot(self.inputs.T, dvalues)
        self.dbiases = sum(dvalues, axis=0, keepdims=True)
        self.dinputs = dot(dvalues, self.weights.T)


class Activation_Softmax_Loss_CategoricalCrossentropy():
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossEntropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        self.loss.calculate(self.output, y_true)
        return self.output
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        self.dinputs = dvalues.copy()
        if len(y_true.shape) == 2:
            self.dinputs[range(samples), y_true] -= 1
            self.dinputs = self.dinputs / samples