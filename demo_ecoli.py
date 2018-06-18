from mlpnn.Factories.MLPNNFactory import MLPNNFactory
from mlpnn.Functions.Sigmoid import Sigmoid
from mlpnn.Import.File import File
from mlpnn.Samples.Samples import Samples
from mlpnn.Structure.MLPNN import MLPNN
from mlpnn.Testing.Accuracy import Accuracy

if __name__ == '__main__':
    file = File('/data/ecoli.csv')
    model = File('/models/ecoli.pkl')

    data = Samples(file, ratio=0.75, shuffle_data=True)

    layers = [data.input_neurons(), 3, 5, data.output_neurons()]

    mlpnn = MLPNNFactory.create(layers, training_strategy=MLPNN.OFFLINE_TRAINING)
    mlpnn.use(Sigmoid()).set_learning_rate(0.25).set_beta(0.5)
    mlpnn.train(data.train_data(), data.train_labels(), epochs=500)
    mlpnn.save(model)

    accuracy = Accuracy.test(mlpnn, data)
    print('Accuracy:', accuracy)
