from sklearn import linear_model
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from Context import Strategy, Context
from sklearn.linear_model import BayesianRidge

class FrameWorkBayesianNetworks(Strategy):

    def __init__(self, model,TrainPath,TestPath,param=None):
        super().__init__(model,"BN",TrainPath,TestPath, param)


    def train(self):
        # super().train()
        self.model.fit(self.x_train, self.y_train)
        self.prediction = self.model.predict(self.x_test)
        r_coefs = self.model.score(self.x_test, self.y_test)
        print(r_coefs)
        # return self.accur

def checkBN():
    reg = linear_model.BayesianRidge()
    model_reg = Context(FrameWorkBayesianNetworks(reg, "trainData26F.csv", "TestData26F.csv"))
    model_reg.run_model()

checkBN()

