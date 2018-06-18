from mlpnn.Factories.LayerFactory import LayerFactory
from mlpnn.Functions.Sigmoid import Sigmoid
from mlpnn.Networks.MLPNN import MLPNN


class MLPNNFactory(object):
    @staticmethod
    def create(neurons_list, use_bias_node=False, training_strategy=MLPNN.ONLINE_TRAINING):
        layers = list()

        if use_bias_node:
            neurons_list[0] += 1

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

        return MLPNN(layers, Sigmoid(), bias_node=use_bias_node, training=training_strategy)
