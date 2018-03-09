from math import exp

from src.Factories.SynapseFactory import SynapseFactory


class Neuron(object):
    def __init__(self, id, learning_rate=0.001, beta=0.05):
        self.id = id
        self.learning_rate = learning_rate
        self.beta = beta
        self.layer = None
        self.input_connections = list()
        self.output_connections = list()
        self.weighted_sum = 0
        self.delta = 0
        self.output = 0

    def belongs_to(self, layer):
        self.layer = layer

    def connect_to(self, neuron):
        synapse = SynapseFactory.create_synapse(self, neuron)

        self.output_connections.append(synapse)
        neuron.input_connections.append(synapse)

    def calculate_sum(self):
        self.weighted_sum = sum(map(lambda synapse: synapse.previous.output * synapse.weight, self.input_connections))

    def calculate_output(self):
        self.output = self._activation_function(self.weighted_sum)

    def set_delta(self, label):
        self.delta = label - self.output

    def calculate_delta(self):
        self.delta = sum(
            map(lambda synapse: synapse.next.delta * synapse.weight * (1 - synapse.next.output) * synapse.next.output,
                self.output_connections))

    def calculate_correction(self, learning_rate=0.05):
        for synapse in self.input_connections:
            weight_correction = learning_rate * self.delta * (1 - self.output) * self.output * synapse.previous.output
            synapse.update_weight(weight_correction)

    def _activation_function(self, x):
        return 1 / (1 + exp(-0.005 * x))
