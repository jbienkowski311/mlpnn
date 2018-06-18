from mlpnn.Abstracts.Model import Model
from mlpnn.Utils.ModelService import ModelService


class AutoEncoder(Model):
    def __init__(self):
        self._service = ModelService()

    @staticmethod
    def load(model_file):
        return ModelService().load(model_file)

    def save(self, model_file):
        self._service.save(model_file, self)
