from src.Factories.MLPFactory import MLPFactory
from src.Utils.File import File
from src.Utils.NormalizedSamples import NormalizedSamples

if __name__ == '__main__':
    file = File('/data_wine.csv')

    training_data = NormalizedSamples(file, kfold_ratio=0.8, shuffle_data=True)
    train_data = training_data.train_data()
    train_labels = training_data.train_labels()

    mlp = MLPFactory.create([len(train_data[0]), 6, 6, len(train_labels[0])])
    mlp.train(train_data, train_labels, epochs=1000)

    for test_data, test_label in zip(training_data.test_data(), training_data.test_labels()):
        predicted_label = mlp.predict(test_data)
        print('Correct:', predicted_label == test_label)
