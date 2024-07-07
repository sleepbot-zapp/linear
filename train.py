from numpy.random import randn
from numpy import mean, argmax
from DenseLayer import Layer_Dense
from LossFunctions import Loss_CategoricalCrossEntropy
from math import inf

def trains(epochs: int, input_x, input_y, *layers: list[Layer_Dense]):
    def_input = input_x.copy()
    loss = inf
    loss_func = Loss_CategoricalCrossEntropy()
    for _ in range(epochs):
        for each_layer in layers:
            each_layer.e_weights = each_layer.weights.copy()
            each_layer.e_biases = each_layer.biases.copy()
            each_layer.weights += 0.05 * randn(*each_layer.weights.shape)
            each_layer.biases += 0.05 * randn(*each_layer.biases.shape)
            input_x = each_layer.forward(input_x)
        temp_loss = loss_func.calculate(input_x, input_y)
        if  temp_loss < loss:
            loss = temp_loss
            predictions = argmax(input_x, axis=1)
            accuracy = mean(predictions==input_y)
            print("loss:", loss)
            print("accuracy:", accuracy*100,"%")
        else:
            each_layer.weights = each_layer.e_weights
            each_layer.biases = each_layer.e_biases
        input_x = def_input