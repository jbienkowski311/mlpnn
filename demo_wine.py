from src.Factories.MLPFactory import MLPFactory
from src.Functions.HyperbolicTangent import HyperbolicTangent
from src.Structure.MLP import MLP
from src.Utils.File import File
from src.Utils.Samples import Samples

if __name__ == '__main__':
    file = File('/data/wine.csv')

    training_data = Samples(file, kfold_ratio=0.8, shuffle_data=True)

    layers = [training_data.input_neurons(), 8, 6, training_data.output_neurons()]

    mlp = MLPFactory.create(layers, training_strategy=MLP.ONLINE_TRAINING)
    mlp.use(HyperbolicTangent()).set_learning_rate(0.5).set_beta(0.5)
    mlp.train(training_data.train_data(), training_data.train_labels(), epochs=250)

    correct_predictions = 0
    for test_data, test_label in zip(training_data.test_data(), training_data.test_labels()):
        predicted_label = mlp.predict(test_data)
        if predicted_label == test_label:
            correct_predictions += 1
        print('Correct:', predicted_label == test_label)
    print('Accuracy:', round(correct_predictions / len(training_data.test_data()) * 100, ndigits=2))
