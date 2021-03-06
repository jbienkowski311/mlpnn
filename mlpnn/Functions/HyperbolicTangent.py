from math import tanh

from mlpnn.Abstracts.Function import Function


class HyperbolicTangent(Function):
    def function(self):
        def inner(x, beta):
            return tanh(beta * x)

        return inner

    def derivative(self):
        def inner(x):
            return 1 - x * x

        return inner
