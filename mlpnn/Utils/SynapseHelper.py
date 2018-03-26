from random import uniform, seed


class SynapseHelper(object):
    START_RANGE = -0.5
    END_RANGE = 0.5

    @staticmethod
    def random_weight():
        seed()
        return uniform(SynapseHelper.START_RANGE, SynapseHelper.END_RANGE)
