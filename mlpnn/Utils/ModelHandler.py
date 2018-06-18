from pickle import dump, load, HIGHEST_PROTOCOL


class ModelHandler(object):
    @staticmethod
    def save(model, file):
        with open(file.path, 'wb') as output:
            dump(model, output, HIGHEST_PROTOCOL)

    @staticmethod
    def load(file):
        with open(file.path, 'rb') as model:
            instance = load(model)

        return instance
