class Accuracy(object):
    @staticmethod
    def test(model, data):
        correct_predictions = 0
        for test_data, test_label in zip(data.test_data(), data.test_labels()):
            predicted_label = model.predict(test_data)
            correct_predictions += int(predicted_label == test_label)

        return round(correct_predictions / len(data.test_data()) * 100, ndigits=2)
