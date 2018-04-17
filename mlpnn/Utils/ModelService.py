from pickle import dump, load, HIGHEST_PROTOCOL


class ModelService(object):
    def save(self, mlpnn, file):
        self._save(mlpnn, file)

    def load(self, file):
        return self._load(file)

    def _save(self, mlpnn, file):
        with open(file.path, 'wb') as output:
            dump(mlpnn, output, HIGHEST_PROTOCOL)

    def _load(self, file):
        with open(file.path, 'rb') as mlpnn:
            instance = load(mlpnn)

        return instance
