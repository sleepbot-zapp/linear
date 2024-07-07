from numpy import maximum


class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = maximum(0, inputs)
        return self.outputs
