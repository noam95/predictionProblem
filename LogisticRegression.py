from sklearn import linear_model, metrics
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import numpy as np
from Context import Strategy, Context
from sklearn.linear_model import LogisticRegression


class FrameWorkLogisticRegression(Strategy):

    def _init_(self, model,TrainPath,TestPath,param=None):
        super()._init_(model, "BN", TrainPath, TestPath, param)
        self.precision = None
        self.recall = None
        self.F_measure = None


    def train(self):
        # super().train()
        self.model.fit(self.x_train, self.y_train)
        self.prediction = self.model.predict(self.x_test)
        self.r_coefs = self.model.score(self.x_test, self.y_test)
        self.accur = metrics.accuracy_score(self.y_test, self.prediction)

    def getFMeasures(self):
        mesures= precision_recall_fscore_support(self.y_test, self.prediction, average='weighted')
        self.recall =mesures[0]
        self.prec = mesures[1]
        self.f1 = mesures[2]

    def getCsvData(self):

        train = self.x_test.columns.values
        columnAsList = list(train)

        #predictions measures

        columns_val = self.x_train.columns.values
        columns_as_list = list(columns_val)
        parameters = self.param
        parameters['numOfF'] = len(columnAsList)
        parameters['num of rows'] = len(self.x_train)
        parameters['C_param'] = self.c_param

        #get all the measures of the model prediction
        self.getFMeasures()
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

def checkBN():
    '''
    The function initialize LR model and run it over the data.
    The results extract to excel file.
    :return:
    '''
    #initialize the logistic regression model
    reg = LogisticRegression()

    #define parameters
    params_values = {"C": np.logspace(-3, 3, 7),
                     "penalty": ["l1", "l2"]}  # l1 lasso l2 ridge

    model = Context(FrameWorkLogisticRegression(reg, "trainData26F.csv", "TestData26F.csv", param=params_values))

    # gridSearch - optional to run the model with
    # model.strategy.grid_search()


    #features selection with k=6 (maintain in the report file)

    model.strategy.x_train = SelectKBest(chi2, k=6).fit_transform(model.strategy.x_train, model.strategy.y_train)
    model.strategy.x_train = pd.DataFrame(model.strategy.x_train)
    model.strategy.x_test = SelectKBest(chi2, k=6).fit_transform(model.strategy.x_test, model.strategy.y_test)
    model.strategy.x_test = pd.DataFrame(model.strategy.x_test)



    model.run_model()

    #get model analyze and write the metrics result to excel
    data = model.strategy.getCsvData()
    model.strategy.insertDataToCSV(data, "reg")


checkBN()