from numpy import exp, sum, max, empty_like, diagflat, dot


class Activation_Softmax:
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = exp(inputs - max(inputs, axis=1, keepdims=True))
        self.output = exp_values / sum(exp_values, axis=1, keepdims=True)
        return self.output

    def backward(self, dvalues):
        self.dinputs = empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = diagflat(single_output) - dot(single_output, single_output.T)
            self.dinputs[index] = dot(jacobian_matrix,single_dvalues)