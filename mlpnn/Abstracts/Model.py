from abc import ABCMeta

from mlpnn.Utils.ModelHandler import ModelHandler


class Model(object):
    ONLINE_TRAINING = 1
    OFFLINE_TRAINING = 2

    __metaclass__ = ABCMeta

    def __init__(self, activation_function, training=ONLINE_TRAINING):
        self.training = training
        self.activation_function = activation_function
        self.beta = 1.0
        self.learning_rate = 0.01
        self.update_learning_rate = False
        self._debug = False

    def use(self, activation_function):
        self.activation_function = activation_function

        return self

    def online_training(self):
        self.training = self.ONLINE_TRAINING

        return self

    def offline_training(self):
        self.training = self.OFFLINE_TRAINING

        return self

    def set_learning_rate(self, learning_rate, update_learning_rate=False):
        self.learning_rate = learning_rate
        # TODO: implement learning rate updating
        self.update_learning_rate = update_learning_rate

        return self

    def set_beta(self, beta):
        self.beta = beta

        return self

    def set_debug(self, debug=True):
        self._debug = debug

        return self

    @staticmethod
    def load(model_file):
        return ModelHandler.load(model_file)

    def save(self, model_file):
        ModelHandler.save(self, model_file)
