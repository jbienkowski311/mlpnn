import os
import unittest

from mlpnn.Factories.LayerFactory import LayerFactory
from mlpnn.Factories.MLPNNFactory import MLPNNFactory
from mlpnn.Functions.Sigmoid import Sigmoid
from mlpnn.Structure.MLPNN import MLPNN
from mlpnn.Import.File import File


class LayerTest(unittest.TestCase):
    def test_mlpnn_creation(self):
        layers = [LayerFactory.create(2), LayerFactory.create(2)]

        mlpnn = MLPNN(layers, Sigmoid())

        self.assertEqual(2, len(mlpnn.layers))

    def test_mlpnn_input_layer_output_layer(self):
        layer1 = LayerFactory.create(2)
        layer2 = LayerFactory.create(4)
        layer3 = LayerFactory.create(3)
        layers = [layer1, layer2, layer3]

        mlpnn = MLPNN(layers, Sigmoid())

        self.assertEqual(layer1, mlpnn.input_layer)
        self.assertEqual(layer3, mlpnn.output_layer)

    def test_mlpnn_structure(self):
        mlpnn = MLPNNFactory.create([2, 3, 2])

        self.assertEqual(2, len(mlpnn.layers[0].neurons))
        self.assertEqual(3, len(mlpnn.layers[1].neurons))
        self.assertEqual(2, len(mlpnn.layers[2].neurons))
        for neuron in mlpnn.layers[0].neurons:
            self.assertEqual(0, len(neuron.input_connections))
            self.assertEqual(3, len(neuron.output_connections))
        for neuron in mlpnn.layers[1].neurons:
            self.assertEqual(2, len(neuron.input_connections))
            self.assertEqual(2, len(neuron.output_connections))
        for neuron in mlpnn.layers[2].neurons:
            self.assertEqual(3, len(neuron.input_connections))
            self.assertEqual(0, len(neuron.output_connections))

    def test_mlpnn_save_load(self):
        mlpnn = MLPNNFactory.create([2, 3, 2], MLPNN.OFFLINE_TRAINING)
        mlpnn.set_beta(2.5)
        mlpnn.set_learning_rate(0.001)
        file = File('/mlpnn.pkl')

        mlpnn.save(file)
        mlpnn_loaded = MLPNN.load(file)

        self.assertTrue(os.path.exists(file.path))
        self.assertTrue(os.path.isfile(file.path))
        self.assertEqual(len(mlpnn.layers), len(mlpnn_loaded.layers))
        for layer, loaded_layer in zip(mlpnn.layers, mlpnn_loaded.layers):
            self.assertEqual(len(layer.neurons), len(loaded_layer.neurons))
            self.assertEqual(len(layer.neurons), len(loaded_layer.neurons))
            for neuron, loaded_neuron in zip(layer.neurons, loaded_layer.neurons):
                self.assertEqual(neuron.id, loaded_neuron.id)
        self.assertEqual(mlpnn.training, mlpnn_loaded.training)
        self.assertEqual(mlpnn.beta, mlpnn_loaded.beta)
        self.assertEqual(mlpnn.learning_rate, mlpnn_loaded.learning_rate)

        os.remove(file.path)


if __name__ == '__main__':
    unittest.main()
