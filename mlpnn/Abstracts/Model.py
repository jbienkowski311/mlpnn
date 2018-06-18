from abc import ABCMeta

from mlpnn.Utils.ModelHandler import ModelHandler


class Model(object):
    ONLINE_TRAINING = 1
    OFFLINE_TRAINING = 2

    __metaclass__ = ABCMeta

    @staticmethod
    def load(model_file):
        return ModelHandler.load(model_file)

    def save(self, model_file):
        ModelHandler.save(self, model_file)
