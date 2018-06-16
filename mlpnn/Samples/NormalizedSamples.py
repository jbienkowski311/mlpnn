from mlpnn.Samples.Samples import Samples


class NormalizedSamples(Samples):
    def data(self):
        _data = super().data()

        return self._normalize(_data)

    def _normalize(self, data):
        for index, sample in enumerate(data):
            sample = self._cast_to_float(sample)
            min_val, max_val = self._get_extremas(sample)
            data[index] = [(i - min_val) / (max_val - min_val) for i in sample]

        return data

    def _get_extremas(self, sample):
        return [min(sample), max(sample)]

    def _cast_to_float(self, sample):
        return list(map(float, sample))
