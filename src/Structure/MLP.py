import operator


class MLP(object):
    ONLINE_TRAINING = 1
    OFFLINE_TRAINING = 2

    def __init__(self, layers, training=ONLINE_TRAINING):
        self.layers = layers
        self.input_layer = layers[0]
        self.output_layer = layers[-1]
        self.training = training

    def train(self, train_data, train_labels, epochs=1):
        for _ in range(epochs):
            for i in range(len(train_data)):
                self._feedforward(train_data[i])
                self._backpropagation(train_labels[i])

            if self.training == self.OFFLINE_TRAINING:
                self._apply_weights_correction()

    def _apply_weights_correction(self):
        for layer in self.layers[:-1]:
            for neuron in layer.neurons:
                neuron.apply_correction()

    def predict(self, input_data):
        self._feedforward(input_data)

        return self.get_output()

    def _feedforward(self, train_data):
        self._set_input(train_data)
        self._calculate_output()

    def _backpropagation(self, train_labels):
        self._set_deltas(train_labels)
        self._correct_weights()

    def get_output(self):
        raw_output = self.get_raw_output()

        index, _ = max(enumerate(raw_output), key=operator.itemgetter(1))
        output = [0] * len(raw_output)
        output[index] = 1

        return output

    def get_raw_output(self):
        output = []

        for neuron in self.output_layer.neurons:
            output.append(neuron.output)

        return output

    def _set_input(self, train_sample):
        for index, value in enumerate(train_sample):
            self.input_layer.neurons[index].output = float(value)

    def _calculate_output(self):
        for index, layer in enumerate(self.layers[1:]):
            for neuron in layer.neurons:
                neuron.calculate_sum()
                neuron.calculate_output()

    def _set_deltas(self, labels):
        for index, label in enumerate(labels):
            self.output_layer.neurons[index].set_delta(float(label))
            self.output_layer.neurons[index].calculate_correction(
                apply_correction=self._should_apply_weight_correction()
            )

    def _should_apply_weight_correction(self):
        return self.training == self.ONLINE_TRAINING

    def _correct_weights(self):
        for layer in reversed(self.layers[:-1]):
            for neuron in layer.neurons:
                neuron.calculate_delta()
                neuron.calculate_correction(
                    apply_correction=self._should_apply_weight_correction()
                )
