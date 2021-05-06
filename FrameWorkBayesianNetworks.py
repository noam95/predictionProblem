from sklearn import linear_model, metrics
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from Context import Strategy, Context
from sklearn.linear_model import BayesianRidge, LinearRegression, LogisticRegression


class FrameWorkBayesianNetworks(Strategy):

    def __init__(self, model,TrainPath,TestPath,param=None):
        super().__init__(model,"BN",TrainPath,TestPath, param)


    def train(self):
        # super().train()
        self.model.fit(self.x_train, self.y_train)
        self.prediction = self.model.predict(self.x_test)
        self.r_coefs = self.model.score(self.x_test, self.y_test)
        self.accur = metrics.accuracy_score(self.y_test, self.prediction)

        #gridSearch
        # if type(self.model) == GridSearchCV:
        #     self.accur = self.model.best_score_

        result =[]
        for pred in self.prediction:
            if pred >= 0: #predict win
                result.append(1)
            else: #predict lose
                result.append(0)
        return self.prediction, result

    def getFMeasures(self):
        mesures= precision_recall_fscore_support(self.y_test, self.prediction, average='weighted')
        self.recall =mesures[0]
        self.prec = mesures[1]
        self.f1 = mesures[2]

    def getCsvData(self):

        train = self.x_test.columns.values
        columnAsList = list(train)

        #predictions measures
        parameters = self.param
        parameters['numOfF'] = len(columnAsList)
        self.getFMeasures()
        # self.get_feature_importance()
        parameters['recall'] = self.recall
        parameters['precision'] = self.prec
        parameters['f1'] = self.f1
        parameters['accuracy'] = self.accur

        #columns names
        colums = list(parameters.keys())
        df = pd.DataFrame(parameters.items()).T
        df.columns =colums
        df = df.iloc[1:]
        return df

    def plotLogisticReg(self, cnf_matrix):
        class_names = ['win', 'lose']  # name  of classes
        fig, ax = plt.subplots()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)
        # create heatmap
        sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
        ax.xaxis.set_label_position("top")
        plt.tight_layout()
        plt.title('Confusion matrix', y=1.1)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.show()

def checkBN():
    reg = LogisticRegression()
    params_values = {"C": np.logspace(-3, 3, 7),
                     "penalty": ["l1", "l2"]}  # l1 lasso l2 ridge

    #features selection
    for i in range(20):
        model = Context(FrameWorkBayesianNetworks(reg, "trainData1.csv", "TestData1.csv", param=params_values))

        model.strategy.x_train = SelectKBest(chi2, k=i+1).fit_transform(model.strategy.x_train, model.strategy.y_train)
        model.strategy.x_train = pd.DataFrame(model.strategy.x_train)
        model.strategy.x_test = SelectKBest(chi2, k=i+1).fit_transform(model.strategy.x_test, model.strategy.y_test)
        model.strategy.x_test = pd.DataFrame(model.strategy.x_test)

        #gridSearch
        # model.strategy.grid_search()

        #train
        model.run_model()

        #getDataToCSV
        data = model.strategy.getCsvData()
        model.strategy.insertDataToCSV(data,"3")



checkBN()

