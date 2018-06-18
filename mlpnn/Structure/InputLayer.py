class InputLayer(object):
    def __init__(self, layer):
        self.layer = layer

    def set_input(self, train_data, bias_node=False):
        for index, value in enumerate(train_data):
            if bias_node and index == 0:
                self.layer.neurons[index].output = 1
                continue
            self.layer.neurons[index].output = float(value)
