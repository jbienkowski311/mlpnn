from mlpnn.Factories.MLPNNFactory import MLPNNFactory
from mlpnn.Functions.Sigmoid import Sigmoid
from mlpnn.Structure.MLPNN import MLPNN
from mlpnn.Import.File import File
from mlpnn.Samples.NormalizedSamples import NormalizedSamples
from mlpnn.Testing.Accuracy import Accuracy

if __name__ == '__main__':
    file = File('/data/iris.csv')
    model = File('/models/iris.pkl')

    data = NormalizedSamples(file, ratio=0.8, shuffle_data=True)

    layers = [data.input_neurons(), 3, 4, data.output_neurons()]

    mlpnn = MLPNNFactory.create(layers, use_bias_node=True, training_strategy=MLPNN.ONLINE_TRAINING)
    mlpnn.use(Sigmoid()).set_learning_rate(0.25).set_beta(0.75)
    mlpnn.train(data.train_data(), data.train_labels(), epochs=750)
    mlpnn.save(model)

    accuracy = Accuracy.test(mlpnn, data)
    print('Accuracy:', accuracy)
