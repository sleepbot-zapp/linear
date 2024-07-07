from numpy import mean


class LossBase:
    def calculate(self, output, y):
        return mean(self.forward(output, y))
