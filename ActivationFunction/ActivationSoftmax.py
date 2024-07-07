from numpy import exp, sum, max


class Activation_Softmax:
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = exp(inputs - max(inputs, axis=1, keepdims=True))
        return exp_values / sum(exp_values, axis=1, keepdims=True)
