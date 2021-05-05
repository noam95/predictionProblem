from sklearn.ensemble import RandomForestClassifier

from Context import Strategy, Context


class FrameWorkRandomForest(Strategy):

    def __init__(self, model, param):
        super().__init__(model,param,"RandomForest")

    def train(self):
        super().train()
        return self.accur

    def removeFeatures(self):
        '''
        optinal: function that remove features with zero importance
        :return:
        '''
        pass




def checkRandomForest():
    RandomForest  = RandomForestClassifier(random_state=0)
    parameter_space = []
    model = Context(FrameWorkRandomForest(RandomForest, param=parameter_space))
    model.strategy.grid_search()
    accurancy_model = model.run_model()
