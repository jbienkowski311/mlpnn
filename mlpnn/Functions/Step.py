from mlpnn.Abstracts.Function import Function


class Step(Function):
    def function(self):
        def inner(x, beta):
            return int(x > 0)

        return inner

    def derivative(self):
        def inner(x):
            return 0

        return inner
