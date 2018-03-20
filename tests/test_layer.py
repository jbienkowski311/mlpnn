import unittest

from src.Factories.LayerFactory import LayerFactory


class LayerTest(unittest.TestCase):
    def setUp(self):
        self.layer = LayerFactory.create(3)

    def test_layer_has_correct_number_of_neurons(self):
        layer = LayerFactory.create(2)

        self.assertEqual(2, len(layer.neurons))

    def test_input_layer_successfully_connected(self):
        layer = LayerFactory.create(2)

        self.layer.connect_input(layer)

        self.assertEqual(1, len(self.layer.input_layers))
        self.assertEqual(0, len(self.layer.output_layers))

    def test_output_layer_successfully_connected(self):
        layer = LayerFactory.create(2)

        self.layer.connect_output(layer)

        self.assertEqual(0, len(self.layer.input_layers))
        self.assertEqual(1, len(self.layer.output_layers))

    def test_neuron_connections_between_layers(self):
        layer = LayerFactory.create(2)

        self.layer.connect_output(layer)

        for neuron in self.layer.neurons:
            self.assertEqual(2, len(neuron.output_connections))


if __name__ == '__main__':
    unittest.main()
