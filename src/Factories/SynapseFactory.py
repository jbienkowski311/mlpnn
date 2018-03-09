from src.Structure.Synapse import Synapse
from src.Utils.SynapseHelper import SynapseHelper


class SynapseFactory(object):
    @staticmethod
    def create_synapse(input_neuron, output_neuron, weight=None):
        if weight is None:
            weight = SynapseHelper.random_weight()

        return Synapse(input_neuron, output_neuron, initial_weight=weight)
