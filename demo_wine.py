from mlpnn.Factories.MLPNNFactory import MLPNNFactory
from mlpnn.Functions.Step import Step
from mlpnn.Import.File import File
from mlpnn.Samples.NormalizedSamples import NormalizedSamples
from mlpnn.Structure.MLPNN import MLPNN
from mlpnn.Testing.Accuracy import Accuracy

if __name__ == '__main__':
    file = File('/data/wine.csv')
    model = File('/models/wine.pkl')

    data = NormalizedSamples(file, ratio=0.8, shuffle_data=True)

    layers = [data.input_neurons(), 8, 4, 6, data.output_neurons()]

    mlpnn = MLPNNFactory.create(layers, use_bias_node=True, training_strategy=MLPNN.ONLINE_TRAINING)
    mlpnn.use(Step()).set_learning_rate(0.3).set_beta(0.6)
    mlpnn.train(data.train_data(), data.train_labels(), epochs=500)
    mlpnn.save(model)

    accuracy = Accuracy.test(mlpnn, data)
    print('Accuracy:', accuracy)
