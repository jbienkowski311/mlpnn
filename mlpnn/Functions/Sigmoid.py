from math import exp

from mlpnn.Abstracts.Function import Function


class Sigmoid(Function):
    def function(self):
        def inner(x, beta):
            return 1 / (1 + exp(-beta * x))

        return inner

    def derivative(self):
        def inner(x):
            return x * (1 - x)

        return inner
