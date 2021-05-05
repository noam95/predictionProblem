from pandas import np
from sklearn.metrics import accuracy_score

from Context import Strategy


class FrameWorkBayesianNetworks(Strategy):

    def __init__(self, model,TrainPath,TestPath,param=None):
        super().__init__(model,"BN",TrainPath,TestPath, param)


    def do_algorithm(self):
        super().train()
        return self.accur

def checkBN():
    pass

