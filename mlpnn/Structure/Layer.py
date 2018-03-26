class Layer(object):
    def __init__(self, neurons):
        self.neurons = neurons
        self.input_layers = list()
        self.output_layers = list()
        self._attach_neurons()

    def connect_input(self, layer):
        self.input_layers.append(layer)

    def connect_output(self, layer):
        self.output_layers.append(layer)
        self._create_connections(layer)

    def _attach_neurons(self):
        for neuron in self.neurons:
            neuron.belongs_to(self)

    def _create_connections(self, layer):
        for neuron in self.neurons:
            for next_layer_neuron in layer.neurons:
                neuron.connect_to(next_layer_neuron)
