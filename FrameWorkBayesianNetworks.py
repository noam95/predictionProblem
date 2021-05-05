from pandas import np
from sklearn.metrics import accuracy_score

from Context import Strategy


class FrameWorkBayesianNetworks(Strategy):

    def __init__(self, model, param):
        super().__init__(model,param,"BN")


    def do_algorithm(self):
        super().train()
        return self.accur

def checkBN():
    pass

