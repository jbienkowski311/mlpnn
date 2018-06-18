from operator import itemgetter


class OutputLayer(object):
    def __init__(self, layer):
        self.layer = layer

    def get_raw_output(self):
        return list(map(lambda neuron: neuron.output, self.layer.neurons))

    def get_output(self):
        raw_output = self.get_raw_output()

        index, _ = max(enumerate(raw_output), key=itemgetter(1))
        output = [0] * len(raw_output)
        output[index] = 1

        return output

    def set_deltas(self, labels, activation_function, apply_correction, learning_rate):
        for index, label in enumerate(labels):
            self.layer.neurons[index].set_delta(float(label))
            self.layer.neurons[index].calculate_correction(
                activation_function.derivative(),
                apply_correction=apply_correction,
                learning_rate=learning_rate
            )
