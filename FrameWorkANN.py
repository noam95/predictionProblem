from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import pandas as pd

from Context import Strategy, Context


class FrameWorkANN(Strategy):
    def __init__(self, model,TrainPath,TestPath,param=None):
        super().__init__(model,"ANN",TrainPath,TestPath, param)
        self.coef_ = None

    def train(self):
        super().train()
        return self.accur
    def getFeatureImportance(self):
        '''
        optional: Feature inmportance
        :return:
        '''
        pass

    def getCsvData(self):

        train = self.x_test.columns.values
        parameters = self.param.items()
        ListTitels = ['accurancy',]
        columnAsList = list(train)
        df = pd.DataFrame(parameters)
        return df


def checkANN():
    clf = MLPClassifier(hidden_layer_sizes=50)
    parameter_space = {
        'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive'],
    }
    model = Context(FrameWorkANN(clf,'trainData26F.csv','TestData26F.csv',param=parameter_space,))
    # model_ann.strategy.grid_search()
    accurancy_model = model.run_model()
    data = model.strategy.getCsvData()
    model.strategy.insertDataToCSV(data, "2")


checkANN()