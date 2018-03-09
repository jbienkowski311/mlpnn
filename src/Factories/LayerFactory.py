from src.Factories.NeuronFactory import NeuronFactory
from src.Structure.Layer import Layer


class LayerFactory(object):
    @staticmethod
    def create(number_of_neurons):
        neurons = list()

        for index in range(number_of_neurons):
            neurons.append(NeuronFactory.create(index + 1))

        return Layer(neurons)
