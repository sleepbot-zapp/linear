from numpy import maximum


class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = maximum(0, inputs)
        return self.outputs

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
        return self.dinputs