from mlpnn.Factories.MLPFactory import MLPFactory
from mlpnn.Functions.Sigmoid import Sigmoid
from mlpnn.Structure.MLP import MLP
from mlpnn.Utils.File import File
from mlpnn.Utils.NormalizedSamples import NormalizedSamples

if __name__ == '__main__':
    file = File('/data/iris.csv')

    training_data = NormalizedSamples(file, kfold_ratio=0.8, shuffle_data=True)

    layers = [training_data.input_neurons(), 4, 3, training_data.output_neurons()]

    mlp = MLPFactory.create(layers, training_strategy=MLP.ONLINE_TRAINING)
    mlp.use(Sigmoid()).set_learning_rate(0.2).set_beta(0.5)
    mlp.train(training_data.train_data(), training_data.train_labels(), epochs=500)

    correct_predictions = 0
    for test_data, test_label in zip(training_data.test_data(), training_data.test_labels()):
        predicted_label = mlp.predict(test_data)
        if predicted_label == test_label:
            correct_predictions += 1
        print('Correct:', predicted_label == test_label)
    print('Accuracy:', round(correct_predictions / len(training_data.test_data()) * 100, ndigits=2))
