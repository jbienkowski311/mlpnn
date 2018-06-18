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

    def feed_forward(self, activation_function, beta):
        for neuron in self.neurons:
            neuron.calculate_sum()
            neuron.calculate_output(activation_function.function(), beta=beta)

    def back_propagate(self, activation_function, apply_correction, learning_rate):
        for neuron in self.neurons:
            neuron.calculate_delta(activation_function.derivative())
            neuron.calculate_correction(
                activation_function.derivative(),
                apply_correction=apply_correction,
                learning_rate=learning_rate
            )

    def _attach_neurons(self):
        for neuron in self.neurons:
            neuron.belongs_to(self)

    def _create_connections(self, layer):
        for neuron in self.neurons:
            for next_layer_neuron in layer.neurons:
                neuron.connect_to(next_layer_neuron)
