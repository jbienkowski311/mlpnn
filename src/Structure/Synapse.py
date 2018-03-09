class Synapse(object):
    def __init__(self, input_neuron, output_neuron, initial_weight=0.0):
        self.previous = input_neuron
        self.next = output_neuron
        self.weight = initial_weight

    def update_weight(self, weight_change):
        self.weight += weight_change
