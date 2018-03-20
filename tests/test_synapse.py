import unittest

from src.Structure.Neuron import Neuron
from src.Structure.Synapse import Synapse


class LayerTest(unittest.TestCase):
    def test_synapse_creation(self):
        neuron1 = Neuron(1)
        neuron2 = Neuron(2)

        synapse = Synapse(neuron1, neuron2, initial_weight=0.5)

        self.assertEqual(neuron1.id, synapse.previous.id)
        self.assertEqual(neuron2.id, synapse.next.id)
        self.assertEqual(0.5, synapse.weight)

    def test_synapse_updates_weight(self):
        neuron1 = Neuron(1)
        neuron2 = Neuron(2)

        synapse = Synapse(neuron1, neuron2, initial_weight=0.5)
        synapse.store_weight(0.5)
        synapse.update_weight()

        self.assertEqual(1.0, synapse.weight)


if __name__ == '__main__':
    unittest.main()
