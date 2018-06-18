from random import uniform, seed


class SynapseHelper(object):
    START_RANGE = -2.0
    END_RANGE = 2.0

    @staticmethod
    def random_weight():
        seed()
        return uniform(SynapseHelper.START_RANGE, SynapseHelper.END_RANGE)
