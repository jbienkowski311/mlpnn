from pickle import dump, load, HIGHEST_PROTOCOL


class ModelService(object):
    def save(self, model, file):
        self._save(model, file)

    def load(self, file):
        return self._load(file)

    def _save(self, model, file):
        with open(file.path, 'wb') as output:
            dump(model, output, HIGHEST_PROTOCOL)

    def _load(self, file):
        with open(file.path, 'rb') as model:
            instance = load(model)

        return instance
