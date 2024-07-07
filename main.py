from nnfs.datasets import spiral_data
from DenseLayer import Layer_Dense
from ActivationFunction import Activation_ReLU, Activation_Softmax
from train import trains

X, y = spiral_data(samples=100, classes=3)

layer1 = Layer_Dense(2, 3, Activation_ReLU)
layer1.forward(X)
layer2 = Layer_Dense(3, 3, Activation_Softmax)
layer2.forward(layer1.output)
trains(10, X, y, layer1, layer2)