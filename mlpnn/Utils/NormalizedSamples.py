from mlpnn.Utils.Samples import Samples


class NormalizedSamples(Samples):
    def data(self):
        _data = super().data()

        return self._normalize(_data)

    def _normalize(self, data):
        for index, sample in enumerate(data):
            data[index] = [(float(i)-float(min(sample)))/(float(max(sample))-float(min(sample))) for i in sample]

        return data
