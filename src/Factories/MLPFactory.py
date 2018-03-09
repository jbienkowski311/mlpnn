from src.Factories.LayerFactory import LayerFactory
from src.Structure.MLP import MLP


class MLPFactory(object):
    @staticmethod
    def create(neurons_list):
        layers = list()

        for number_of_neurons in neurons_list:
            layers.append(LayerFactory.create(number_of_neurons))

        for index in range(len(neurons_list)):
            if index == 0:
                layers[index].connect_output(layers[index + 1])
            elif index == len(neurons_list) - 1:
                layers[index].connect_input(layers[index - 1])
            else:
                layers[index].connect_input(layers[index - 1])
                layers[index].connect_output(layers[index + 1])

        return MLP(layers)
