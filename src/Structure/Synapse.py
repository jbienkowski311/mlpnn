class Synapse(object):
    def __init__(self, input_neuron, output_neuron, initial_weight=0.0):
        self.previous = input_neuron
        self.next = output_neuron
        self.weight = initial_weight
        self.weights = list()

    def store_weight(self, weight):
        self.weights.append(weight)

    def update_weight(self):
        self.weight += sum(self.weights) / float(len(self.weights))
        self.weights = list()
