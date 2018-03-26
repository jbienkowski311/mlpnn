import unittest

from mlpnn.Factories.LayerFactory import LayerFactory
from mlpnn.Factories.MLPFactory import MLPFactory
from mlpnn.Functions.Sigmoid import Sigmoid
from mlpnn.Structure.MLP import MLP


class LayerTest(unittest.TestCase):
    def test_mlp_creation(self):
        layers = [LayerFactory.create(2), LayerFactory.create(2)]

        mlp = MLP(layers, Sigmoid())

        self.assertEqual(2, len(mlp.layers))

    def test_mlp_input_layer_output_layer(self):
        layer1 = LayerFactory.create(2)
        layer2 = LayerFactory.create(4)
        layer3 = LayerFactory.create(3)
        layers = [layer1, layer2, layer3]

        mlp = MLP(layers, Sigmoid())

        self.assertEqual(layer1, mlp.input_layer)
        self.assertEqual(layer3, mlp.output_layer)

    def test_mlp_structure(self):
        mlp = MLPFactory.create([2, 3, 2])

        self.assertEqual(2, len(mlp.layers[0].neurons))
        self.assertEqual(3, len(mlp.layers[1].neurons))
        self.assertEqual(2, len(mlp.layers[2].neurons))
        for neuron in mlp.layers[0].neurons:
            self.assertEqual(0, len(neuron.input_connections))
            self.assertEqual(3, len(neuron.output_connections))
        for neuron in mlp.layers[1].neurons:
            self.assertEqual(2, len(neuron.input_connections))
            self.assertEqual(2, len(neuron.output_connections))
        for neuron in mlp.layers[2].neurons:
            self.assertEqual(3, len(neuron.input_connections))
            self.assertEqual(0, len(neuron.output_connections))


if __name__ == '__main__':
    unittest.main()
