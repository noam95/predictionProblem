from sklearn import linear_model, metrics
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
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
        r_coefs = self.model.score(self.x_test, self.y_test)
        result =[]
        for pred in self.prediction:
            if pred >= 0: #predict win
                result.append(1)
            else: #predict lose
                result.append(0)
        self.accur = metrics.accuracy_score(self.y_test, self.prediction)
        y_test_list = list(self.y_test)
        counter = 0
        # for i in range(len(result)):
        #     if result[i] == y_test_list[i]:
        #         counter += 1
        # accur = counter/len(result)
        return self.prediction, result

    def getCsvData(self):

        columns_val = self.x_train.columns.values
        columns_as_list = list(columns_val)
        parameters = self.param
        parameters['accuracy'] = self.accur
        parameters['numOfFeatures'] = len(columns_as_list)
        colums = list(parameters.keys())
        df = pd.DataFrame(parameters.items()).T
        df.columns = colums
        df = df.iloc[1:]
        return df

def checkBN():
    reg = LogisticRegression()

    #normalize the data
    # scaler = StandardScaler()
    # model_reg.strategy.x_train = scaler.fit_transform(model_reg.strategy.x_train)
    # model_reg.strategy.x_test = scaler.fit_transform(model_reg.strategy.x_test)
    params_values = {
        'panalty': 'l2',
        'dual': 'False',
        'tol': '0.001',
        'C': '0.8',
        'solver': 'liblinear'

    }
    model = Context(FrameWorkBayesianNetworks(reg, "trainData26F.csv", "TestData26F.csv", param=params_values))
    y_pred, result = model.run_model()
    print("Accurancy" + str(metrics.accuracy_score(model.strategy.y_test, y_pred)))
    # parameter_space = {
    #     'hidden_layer_sizes': [(5,10,5),(15,),(5,5,5),(8,8),(100,5,100)],
    #     'activation': ['tanh', 'relu'],
    #     # 'solver': ['sgd', 'adam'],
    #     'alpha': [0.0001, 0.05],
    #     # 'learning_rate': ['constant', 'invscaling', 'adaptive'],
    #     'max_iter':[1,200]
    # }

    # param_values = {penalty= 'l2', dual= False,tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None
    # }

    model.strategy.x_train = SelectKBest(chi2, k=8).fit_transform(model.strategy.x_train, model.strategy.y_train)
    model.strategy.x_train = pd.DataFrame(model.strategy.x_train)
    model.strategy.x_test = SelectKBest(chi2, k=8).fit_transform(model.strategy.x_test, model.strategy.y_test)
    model.strategy.x_test = pd.DataFrame(model.strategy.x_test)
    # model.strategy.grid_search()
    y_pred, result = model.run_model()
    data = model.strategy.getCsvData()
    model.strategy.insertDataToCSV(data,"1")

    print(result)


checkBN()

