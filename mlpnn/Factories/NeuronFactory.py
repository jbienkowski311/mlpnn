from mlpnn.Structure.Neuron import Neuron


class NeuronFactory(object):
    @staticmethod
    def create(id):
        return Neuron(id)
