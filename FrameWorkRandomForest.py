from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from Context import Strategy, Context


class FrameWorkRandomForest(Strategy):

    def __init__(self, model,TrainPath,TestPath,param=None):
        super().__init__(model,"RanfomForest",TrainPath,TestPath, param)

    def train(self):
        super().train()
        return self.accur

    def removeFeatures(self):
        '''
        optinal: function that remove features with zero importance
        :return:
        '''
        pass
    def get_feature_importance(self, model):
        return self.model.feature_importances_


def checkRandomForest():
    RandomForest = RandomForestClassifier(random_state=0)
    parameter_space = []
    model = Context(FrameWorkRandomForest(RandomForest, "TestData26F.csv", "trainData26F.csv"))
    accurancy_model = model.run_model()
    importance = RandomForest.feature_importances_
    train = model.strategy.x_train
    columns = train.columns.values
    data_df = {'Model name': ["Random Forest"],
                'Number of features': [len(columns)],
                'Accurancy': [model.strategy.accur]
                }
    i = 0
    for col in columns[0:len(train.columns) - 1]:
        data_df[col] = importance[i]
        i += 1

    df = pd.DataFrame(data_df)
    model.strategy.insertDataToCSV(data_df)


checkRandomForest()