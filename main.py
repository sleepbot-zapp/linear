from nnfs.datasets import spiral_data
from DenseLayer import Layer_Dense, Activation_Softmax_Loss_CategoricalCrossentropy
from ActivationFunction import Activation_ReLU, Activation_Softmax
from train import trains

X, y = spiral_data(samples=100, classes=3)

layer1 = Layer_Dense(2, 3, Activation_ReLU)
layer1.forward(X)
layer2 = Layer_Dense(3, 3, Activation_Softmax_Loss_CategoricalCrossentropy)
layer2.forward(layer1.output, y_true=y)
# trains(10, X, y, layer1, layer2)
# print('acc:', accuracy)

loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
loss_activation.backward(layer2.output, y)
layer2.backward(loss_activation.dinputs)
layer1.activation_function_backward(layer2.dinputs)
layer1.backward(layer1.dinputs)
