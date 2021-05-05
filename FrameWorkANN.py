from sklearn.feature_selection import SelectKBest, chi2
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
        columnAsList = list(train)

        parameters = self.param
        parameters['accurancy'] = self.accur
        parameters['numOfF'] = len(columnAsList)
        colums = list(parameters.keys())
        df = pd.DataFrame(parameters.items()).T
        df.columns =colums
        df = df.iloc[1:]
        return df


def checkANN():



    clf = MLPClassifier()
    parameter_space = {
        'hidden_layer_sizes': [(5,10,5),(15,),(5,5,5),(8,8),(100,5,100)],
        'activation': ['tanh', 'relu'],
        # 'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        # 'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'max_iter':[1,200]
    }
    model = Context(FrameWorkANN(clf,'trainData26F.csv','TestData26F.csv',param=parameter_space,))
    model.strategy.x_train = SelectKBest(chi2, k=8).fit_transform(model.strategy.x_train, model.strategy.y_train)
    model.strategy.x_train = pd.DataFrame(model.strategy.x_train)
    model.strategy.x_test = SelectKBest(chi2, k=8).fit_transform(model.strategy.x_test, model.strategy.y_test)
    model.strategy.x_test = pd.DataFrame(model.strategy.x_test)
    model.strategy.grid_search()
    model.run_model()
    data = model.strategy.getCsvData()
    model.strategy.insertDataToCSV(data, "2")


checkANN()