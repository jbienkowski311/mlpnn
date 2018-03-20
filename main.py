from src.Factories.MLPFactory import MLPFactory
from src.Structure.MLP import MLP
from src.Utils.File import File
from src.Utils.NormalizedSamples import NormalizedSamples

if __name__ == '__main__':
    file = File('/data/iris.csv')

    training_data = NormalizedSamples(file, kfold_ratio=0.8, shuffle_data=True)

    layers = [training_data.input_neurons(), 4, 3, training_data.output_neurons()]

    mlp = MLPFactory.create(layers, training_strategy=MLP.ONLINE_TRAINING)
    mlp.train(training_data.train_data(), training_data.train_labels(), epochs=1000)

    for test_data, test_label in zip(training_data.test_data(), training_data.test_labels()):
        predicted_label = mlp.predict(test_data)
        print('Correct:', predicted_label == test_label)
