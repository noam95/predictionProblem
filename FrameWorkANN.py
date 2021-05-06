from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import pandas as pd
from matplotlib import pyplot as plt

from Context import Strategy, Context


class FrameWorkANN(Strategy):
    def __init__(self, model,TrainPath,TestPath,param=None):
        super().__init__(model,"ANN",TrainPath,TestPath, param)
        self.coef_ = None
        self.recall = None
        self.prec = None
        self.f1 = None
        self.activation =None
        self.solver = None
        self.alpha = None
        self.learning_rate= None
        self.max_iter = None


    def train(self):
        super().train()
        return self.accur

    def get_feature_importance(self):
            # top =2
        train = self.x_test.columns.values
        features_names = list(train)
        # features_names = ['defencePressure', 'buildUpPlaySpeed', 'buildUpPlayPassing', 'chanceCreationPassing',
        #                   'chanceCreationCrossing',
        #                   'chanceCreationShooting', 'defencePressure', 'defenceAggression', 'defenceTeamWidth',
        #                   'crossing', 'finishing', 'heading_accuracy', 'volleys', 'dribbling', 'curve',
        #                   'long_passing',
        #                   'aggression', 'short_passing', 'potential', 'overall_rating', 'long_shots',
        #                   'ball_control']
        imp = self.model.coef_[0]
        imp, features_names = zip(*sorted(zip(imp, features_names)))
        plt.barh(range(len(features_names)), imp, align='center')
        plt.yticks(range(len(features_names)), features_names)
        # plt.barh(range(top), imp[::-1][0:top], align='center')
        # plt.yticks(range(top), features_names[::-1][0:top])
        plt.show()

    def getFMeasures(self):
        mesures= precision_recall_fscore_support(self.y_test, self.prediction, average='weighted')
        self.recall =mesures[0]
        self.prec = mesures[1]
        self.f1 = mesures[2]

    def getCsvData(self):

        train = self.x_test.columns.values
        columnAsList = list(train)

        #predictions measures
        parameters ={}# self.param
        parameters['numOfF'] = len(columnAsList)
        self.getFMeasures()
        # self.get_feature_importance()
        parameters['recall'] = self.recall
        parameters['precision'] = self.prec
        parameters['f1'] = self.f1
        parameters['accuracy'] = self.accur
        parameters['activation'] = self.activation
        parameters['solver'] = self.solver
        parameters['alpha'] = self.alpha
        parameters['learning_rate'] = self.learning_rate
        parameters['max_iter'] = self.max_iter


        #columns names
        colums = list(parameters.keys())
        df = pd.DataFrame(parameters.items()).T
        df.columns =colums
        df = df.iloc[1:]
        return df


def checkANN():
    activation = 'relu'
    solver = 'adam'
    alpha = 0.05
    learning_rate = 'constant'
    max_iter = 500

    clf = MLPClassifier(activation=activation,solver=solver,alpha=alpha,learning_rate=learning_rate,max_iter=max_iter)
    parameter_space = {
        # 'hidden_layer_sizes': [(5,10,5),(15,),(5,5,5),(8,8),(100,5,100)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.5],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'max_iter':[50,500]
    }

    model = Context(FrameWorkANN(clf, "trainData1.csv", "TestData1.csv", param=parameter_space))

    model.strategy.activation = activation
    model.strategy.solver = solver
    model.strategy.alpha = alpha
    model.strategy.learning_rate = learning_rate
    model.strategy.max_iter = max_iter
    model.strategy.x_train = SelectKBest(chi2, k=14).fit_transform(model.strategy.x_train, model.strategy.y_train)
    model.strategy.x_train = pd.DataFrame(model.strategy.x_train)
    model.strategy.x_test = SelectKBest(chi2, k=14).fit_transform(model.strategy.x_test, model.strategy.y_test)
    model.strategy.x_test = pd.DataFrame(model.strategy.x_test)
    # model.strategy.grid_search()
    model.run_model()
    data = model.strategy.getCsvData()
    model.strategy.insertDataToCSV(data, "decreseRows")

checkANN()