from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from Context import Strategy, Context
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support

class FrameWorkRandomForest(Strategy):

    def __init__(self, model,TrainPath,TestPath,param=None):
        super().__init__(model,"RanfomForest", TrainPath,TestPath, param)
        self.recall = None
        self.prec = None
        self.f1 = None

    def train(self):
        super().train()
        return self.accur
    def metrics(self):
        scores = precision_recall_fscore_support(self.y_test, self.prediction, average='weighted')
        self.prec = scores[0]
        self.recall = scores[1]
        self.f1 = scores[2]


    def removeFeatures(self):
        '''
        optinal: function that remove features with zero importance
        :return:
        '''
        pass
    def get_feature_importance(self, model):
        return self.model.feature_importances_

    def plot_feature_importance(self,importance, names, model_type):
        # Create arrays from feature importance and feature names
        feature_importance = np.array(importance)
        feature_names = np.array(names)

        # Create a DataFrame using a Dictionary
        data = {'feature_names': feature_names, 'feature_importance': feature_importance}
        fi_df = pd.DataFrame(data)

        # Sort the DataFrame in order decreasing feature importance
        fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

        # Define size of bar plot
        plt.figure(figsize=(10, 8))
        # Plot Searborn bar chart
        sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
        # Add chart labels
        plt.title(model_type + 'FEATURE IMPORTANCE')
        plt.xlabel('FEATURE IMPORTANCE')
        plt.ylabel('FEATURE NAMES')
        plt.show()



def checkRandomForest():
    RandomForest = RandomForestClassifier(random_state=0)
    parameter_space = []
    model = Context(FrameWorkRandomForest(RandomForest, "train_data.csv", "test_data.csv"))
    accurancy_model = model.run_model()
    importance = RandomForest.feature_importances_
    train = model.strategy.x_train
    columns = train.columns.values
    model.strategy.metrics()
    data_df = {'Model name': ["Random Forest"],
                'Number of features': [len(columns)],
                'Accuracy': [model.strategy.accur],
                'Precision': [model.strategy.prec],
                'Recall': [model.strategy.recall],
                'Fscore': [model.strategy.f1]
                }
    i = 0
    for col in columns[0:len(train.columns) - 1]:
        data_df[col] = importance[i]
        i += 1

    df = pd.DataFrame(data_df)
    model.strategy.insertDataToCSV(df,"16records")
    model.strategy.plot_feature_importance(importance, columns, 'RANDOM FOREST')

checkRandomForest()