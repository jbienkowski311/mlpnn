from mlpnn.Factories.MLPNNFactory import MLPNNFactory
from mlpnn.Functions.HyperbolicTangent import HyperbolicTangent
from mlpnn.Structure.MLPNN import MLPNN
from mlpnn.Import.File import File
from mlpnn.Samples.Samples import Samples

if __name__ == '__main__':
    file = File('/data/wine.csv')

    data = Samples(file, kfold_ratio=0.8, shuffle_data=True)

    layers = [data.input_neurons(), 8, 6, data.output_neurons()]

    mlpnn = MLPNNFactory.create(layers, training_strategy=MLPNN.ONLINE_TRAINING)
    mlpnn.use(HyperbolicTangent()).set_learning_rate(0.5).set_beta(0.5)
    mlpnn.train(data.train_data(), data.train_labels(), epochs=250)

    accuracy = Accuracy.test(mlpnn, data)
    print('Accuracy:', accuracy)
