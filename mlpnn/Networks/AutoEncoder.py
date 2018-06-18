from mlpnn.Abstracts.Model import Model
from mlpnn.Structure.InputLayer import InputLayer
from mlpnn.Structure.OutputLayer import OutputLayer


class AutoEncoder(Model):
    def __init__(self, layers, activation_function, training=Model.ONLINE_TRAINING):
        super().__init__(activation_function, training)
        self.layers = layers
        self.input_layer = InputLayer(layers[0])
        self.output_layer = OutputLayer(layers[-1])
        self._last_sample = False

    def train(self, train_data, train_labels, epochs=1):
        for _ in range(epochs):
            for i in range(len(train_data)):
                self._last_sample = i == (len(train_data) - 1)
                self._feed_forward(train_data[i])
                self._back_propagate(train_labels[i])

    def _feed_forward(self, train_data):
        self._set_input(train_data)
        self._calculate_output()

    def _back_propagate(self, train_labels):
        self._set_deltas(train_labels)
        self._correct_weights()

    def _set_input(self, train_data):
        self.input_layer.set_input(train_data)

    def _calculate_output(self):
        for layer in self.layers[1:]:
            layer.feed_forward(self.activation_function, self.beta)

    def _set_deltas(self, labels):
        self.output_layer.set_deltas(
            labels, self.activation_function, self._should_apply_correction(), self.learning_rate
        )

    def _correct_weights(self):
        for layer in reversed(self.layers[:-1]):
            layer.back_propagate(self.activation_function, self._should_apply_correction(), self.learning_rate)

    def _should_apply_correction(self):
        if self.training == self.OFFLINE_TRAINING:
            return self._last_sample

        return self.training == self.ONLINE_TRAINING
