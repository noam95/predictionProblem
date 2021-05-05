from sklearn import linear_model, metrics
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
        y_test_list = list(self.y_test)
        counter = 0
        for i in range(len(result)):
            if result[i] == y_test_list[i]:
                counter += 1
        accur = counter/len(result)
        return self.prediction, result, accur

def checkBN():

    # reg = linear_model.BayesianRidge()
    reg = LogisticRegression()

    # reg = LinearRegression()
    model_reg = Context(FrameWorkBayesianNetworks(reg, "trainData26F.csv", "TestData26F.csv"))

    #normalize the data
    # scaler = StandardScaler()
    # model_reg.strategy.x_train = scaler.fit_transform(model_reg.strategy.x_train)
    # model_reg.strategy.x_test = scaler.fit_transform(model_reg.strategy.x_test)

    y_pred, result, accur = model_reg.run_model()
    print("Accurancy" + str(metrics.accuracy_score(model_reg.strategy.y_test, y_pred)))


    print(result)


checkBN()

