from mlpnn.Factories.LayerFactory import LayerFactory
from mlpnn.Functions.Sigmoid import Sigmoid
from mlpnn.Structure.MLPNN import MLPNN


class MLPNNFactory(object):
    @staticmethod
    def create(neurons_list, training_strategy=MLPNN.ONLINE_TRAINING):
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

        return MLPNN(layers, Sigmoid(), training=training_strategy)
