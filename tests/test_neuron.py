import unittest
from random import randint

from src.Factories.NeuronFactory import NeuronFactory
from src.Structure.Layer import Layer


class NeuronTest(unittest.TestCase):
    def setUp(self):
        self.neuron = NeuronFactory.create(1)

    def test_neuron_connected_to_layer(self):
        layer = Layer([self.neuron])

        self.assertEqual(self.neuron.layer, layer)

    def test_neuron_calculates_weighted_sum(self):
        neuron1 = NeuronFactory.create(1)
        neuron2 = NeuronFactory.create(1)
        neuron1.output = 1
        neuron2.output = 1
        neuron1.connect_to(self.neuron)
        neuron2.connect_to(self.neuron)
        self.neuron.input_connections[0].weight = 1.0
        self.neuron.input_connections[1].weight = 1.0

        self.neuron.calculate_sum()

        self.assertEqual(2.0, self.neuron.weighted_sum)

    def test_neuron_sets_delta(self):
        self.neuron.output = 0.5

        self.neuron.set_delta(1.0)

        self.assertEqual(0.5, self.neuron.delta)

    def test_neuron_calculates_delta(self):
        neuron = NeuronFactory.create(1)
        self.neuron.connect_to(neuron)
        neuron.output = 0.5
        neuron.delta = 0.2
        self.neuron.output_connections[0].weight = 0.1

        self.neuron.calculate_delta()

        self.assertEqual(0.005, round(self.neuron.delta, ndigits=5))


if __name__ == '__main__':
    unittest.main()
