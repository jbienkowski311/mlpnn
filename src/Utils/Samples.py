from src.Utils.InputHandler import InputHandler


class Samples(object):
    def __init__(self, file, kfold_ratio=1.0, shuffle_data=False):
        self.ratio = kfold_ratio
        self.raw_data = InputHandler.handle(file, shuffle_data)

    def data(self):
        return list(map(lambda data: data[:-1], self.raw_data))

    def labels(self):
        return list(map(lambda data: self._label(data[-1]).pop()['output'], self.raw_data))

    def input_neurons(self):
        return len(self.raw_data[-1])

    def output_neurons(self):
        return len(self.labels()[-1])

    def train_data(self):
        _data = self.data()

        return _data[:round(self.ratio*len(_data))]

    def train_labels(self):
        _labels = self.labels()

        return _labels[:round(self.ratio * len(_labels))]

    def test_data(self):
        _data = self.data()

        return _data[round(self.ratio * len(_data)):]

    def test_labels(self):
        _labels = self.labels()

        return _labels[round(self.ratio * len(_labels)):]

    def classes(self):
        unique_names = sorted(set(self._class_names()))
        _classes = list()

        for index, class_name in enumerate(unique_names):
            one_hot = [0] * len(unique_names)
            one_hot[index] = 1
            _classes.append({'name': class_name, 'output': one_hot})

        return _classes

    def _label(self, class_name):
        return list(filter(lambda name: name['name'] == class_name, self.classes()))

    def _class_names(self):
        return list(map(lambda data: data[-1], self.raw_data))
