from sklearn import linear_model, metrics
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils.fixes import loguniform

from Context import Strategy, Context
from sklearn.linear_model import BayesianRidge, LinearRegression, LogisticRegression


class FrameWorkBayesianNetworks(Strategy):

    def __init__(self, model,TrainPath,TestPath,param=None):
        super().__init__(model, "BN", TrainPath, TestPath, param)
        self.precision = None
        self.recall = None
        self.F_measure = None


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
        self.precision = metrics.precision_score(self.y_test, self.prediction)
        self.recall = metrics.recall_score(self.y_test, self.prediction)
        self.F_measure = metrics.f1_score(self.y_test, self.prediction)
        return self.prediction, result

    def getCsvData(self):
        columns_val = self.x_train.columns.values
        columns_as_list = list(columns_val)
        parameters = self.param
        parameters['accuracy'] = self.accur
        parameters['precision'] = self.precision
        parameters['recall'] = self.recall
        parameters['f1-score'] = self.F_measure
        parameters['numOfFeatures'] = len(columns_as_list)
        colums = list(parameters.keys())
        df = pd.DataFrame(parameters.items()).T
        df.columns = colums
        df = df.iloc[1:]
        return df

def checkBN():
    reg = LogisticRegression()
    # params_values = {'C': [loguniform(1e0, 1e3)],
    #     'gamma': [loguniform(1e-4, 1e-3)],
    #     'kernel': ['rbf'],
    #     'class_weight':['balanced', None]}
    # grid_values = {'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    solver_options = ['newton-cg', 'lbfgs', 'liblinear', 'sag']
    multi_class_options = ['ovr']
    class_weight_options = ['None', 'balanced']

    param_grid = dict(solver=solver_options, multi_class= multi_class_options, class_weight=class_weight_options)
    model = Context(FrameWorkBayesianNetworks(reg, "trainData26F.csv", "TestData26F.csv", param=param_grid))
    # grid = GridSearchCV(reg, param_grid, cv=12, scoring= 'accuracy')
    # grid.fit(model.strategy.x_train,model.strategy.y_train)
    # pred = grid.predict(model.strategy.x_test)
    # print("accur " + metrics.accuracy_score(model.strategy.y_test, model.strategy.prediction))
    # # model.strategy.model.estimator.get_params().keys()
    # y_pred, result = model.run_model()
    # print("Accurancy" + str(metrics.accuracy_score(model.strategy.y_test, y_pred)))


    # param_values = {penalty= 'l2', dual= False,tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None
    # }
    index = 21
    while index >= 1:
        model.strategy.x_train = SelectKBest(chi2, k=index).fit_transform(model.strategy.x_train, model.strategy.y_train)
        model.strategy.x_train = pd.DataFrame(model.strategy.x_train)
        model.strategy.x_test = SelectKBest(chi2, k=index).fit_transform(model.strategy.x_test, model.strategy.y_test)
        model.strategy.x_test = pd.DataFrame(model.strategy.x_test)
        # model.strategy.grid_search()
        y_pred, result = model.run_model()
        data = model.strategy.getCsvData()
        model.strategy.insertDataToCSV(data, "1")
        index -= 1

checkBN()

