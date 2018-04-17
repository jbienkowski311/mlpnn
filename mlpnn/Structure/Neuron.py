from mlpnn.Factories.SynapseFactory import SynapseFactory


class Neuron(object):
    def __init__(self, id):
        self.id = id
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
        self.weighted_sum = sum(
            map(
                lambda synapse: synapse.previous.output * synapse.weight, self.input_connections
            )
        )

    def calculate_output(self, activation_function, beta=0.005):
        self.output = activation_function(self.weighted_sum, beta)

    def set_delta(self, label):
        self.delta = label - self.output

    def calculate_delta(self, derivative_function):
        self.delta = sum(
            map(
                lambda synapse:
                synapse.next.delta * synapse.weight * derivative_function(synapse.next.output),
                self.output_connections
            )
        )

    def calculate_correction(self, derivative_function, apply_correction=False, learning_rate=0.05):
        for synapse in self.input_connections:
            weight_correction = learning_rate * self.delta * derivative_function(self.output) * synapse.previous.output
            synapse.store_weight(weight_correction)
            if apply_correction:
                synapse.update_weight()
