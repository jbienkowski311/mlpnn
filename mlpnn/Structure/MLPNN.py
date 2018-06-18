from mlpnn.Abstracts.Model import Model
from mlpnn.Structure.InputLayer import InputLayer
from mlpnn.Structure.OutputLayer import OutputLayer


class MLPNN(Model):
    def __init__(self, layers, activation_function, bias_node=False, training=Model.ONLINE_TRAINING):
        self.bias_node = bias_node
        self.layers = layers
        self.input_layer = InputLayer(layers[0])
        self.output_layer = OutputLayer(layers[-1])
        self.training = training
        self.activation_function = activation_function
        self.beta = 1.0
        self.learning_rate = 0.01
        self.update_learning_rate = False
        self._debug = False
        self._last_sample = False

    def use(self, activation_function):
        self.activation_function = activation_function

        return self

    def online_training(self):
        self.training = self.ONLINE_TRAINING

        return self

    def offline_training(self):
        self.training = self.OFFLINE_TRAINING

        return self

    def set_learning_rate(self, learning_rate, update_learning_rate=False):
        self.learning_rate = learning_rate
        # TODO: implement learning rate updating
        self.update_learning_rate = update_learning_rate

        return self

    def set_beta(self, beta):
        self.beta = beta

        return self

    def debug(self):
        self._debug = True

        return self

    def train(self, train_data, train_labels, epochs=1):
        for _ in range(epochs):
            for i in range(len(train_data)):
                self._last_sample = i == (len(train_data) - 1)
                self._feedforward(train_data[i])
                self._backpropagation(train_labels[i])

    def predict(self, input_data):
        self._feedforward(input_data)

        return self.get_output()

    def get_output(self):
        return self.output_layer.get_output()

    def get_raw_output(self):
        return self.output_layer.get_raw_output()

    def _feedforward(self, train_data):
        self._set_input(train_data)
        self._calculate_output()

    def _backpropagation(self, train_labels):
        self._set_deltas(train_labels)
        self._correct_weights()

    def _set_input(self, train_data):
        self.input_layer.set_input(train_data, self.bias_node)

    def _calculate_output(self):
        for index, layer in enumerate(self.layers[1:]):
            for neuron in layer.neurons:
                neuron.calculate_sum()
                neuron.calculate_output(self.activation_function.function(), beta=self.beta)

    def _set_deltas(self, labels):
        self.output_layer.set_deltas(
            labels, self.activation_function, self._should_apply_correction(), self.learning_rate
        )

    def _correct_weights(self):
        for layer in reversed(self.layers[:-1]):
            for neuron in layer.neurons:
                neuron.calculate_delta(
                    self.activation_function.derivative()
                )
                neuron.calculate_correction(
                    self.activation_function.derivative(),
                    apply_correction=self._should_apply_correction(),
                    learning_rate=self.learning_rate
                )

    def _should_apply_correction(self):
        if self.training == self.OFFLINE_TRAINING:
            return self._last_sample

        return self.training == self.ONLINE_TRAINING
