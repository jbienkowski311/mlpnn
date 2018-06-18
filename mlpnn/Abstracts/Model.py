from abc import ABCMeta, abstractmethod


class Model(object):
    ONLINE_TRAINING = 1
    OFFLINE_TRAINING = 2

    __metaclass__ = ABCMeta

    @staticmethod
    @abstractmethod
    def load(model_file):
        pass

    @abstractmethod
    def save(self, model_file):
        pass
