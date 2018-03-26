from abc import ABCMeta, abstractmethod


class Function(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def function(self):
        pass

    @abstractmethod
    def derivative(self):
        pass
