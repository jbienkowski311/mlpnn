from abc import ABCMeta, abstractmethod


class Model(object):
    __metaclass__ = ABCMeta

    @staticmethod
    @abstractmethod
    def load(model_file):
        pass

    @abstractmethod
    def save(self, model_file):
        pass
