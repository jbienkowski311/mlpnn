class LayerHelper(object):
    @staticmethod
    def get_synapses(layer, input=False):
        if input:
            return LayerHelper.get_input_synapses(layer)

        return LayerHelper.get_output_synapses(layer)

    @staticmethod
    def get_input_synapses(layer):
        if layer is None:
            return None

        return map(lambda neuron: neuron.input_connections, layer.neurons)

    @staticmethod
    def get_output_synapses(layer):
        if layer is None:
            return None

        return map(lambda neuron: neuron.output_connections, layer.neurons)
