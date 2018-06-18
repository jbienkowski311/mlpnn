from mlpnn.Import.CsvImporter import InputHandler


class Data(object):
    def __init__(self, file, shuffle_data=False):
        self.raw_data = InputHandler.handle(file, shuffle_data)

    def classes(self):
        unique_names = sorted(set(self._class_names()))
        _classes = list()

        for index, class_name in enumerate(unique_names):
            one_hot = [0] * len(unique_names)
            one_hot[index] = 1
            _classes.append({'name': class_name, 'output': one_hot})

        return _classes

    def data(self):
        return list(map(lambda data: data[:-1], self.raw_data))

    def labels(self):
        return list(map(lambda data: self._label(data[-1]).pop()['output'], self.raw_data))

    def samples_count(self):
        return len(self.raw_data[-1])

    def labels_count(self):
        return len(self.labels()[-1])

    def _label(self, class_name):
        return list(filter(lambda name: name['name'] == class_name, self.classes()))

    def _class_names(self):
        return list(map(lambda data: data[-1], self.raw_data))
