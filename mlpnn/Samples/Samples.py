from mlpnn.Import.Data import Data


class Samples(object):
    def __init__(self, file, ratio=1.0, shuffle_data=False):
        self.ratio = ratio
        self.data = Data(file, shuffle_data)

    def input_neurons(self):
        return self.data.samples_count() - 1

    def output_neurons(self):
        return self.data.labels_count()

    def train_data(self):
        _data = self.data.data()

        return _data[:round(self.ratio * len(_data))]

    def train_labels(self):
        _labels = self.data.labels()

        return _labels[:round(self.ratio * len(_labels))]

    def test_data(self):
        _data = self.data.data()

        return _data[round(self.ratio * len(_data)):]

    def test_labels(self):
        _labels = self.data.labels()

        return _labels[round(self.ratio * len(_labels)):]
