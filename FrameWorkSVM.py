from pyexpat import features

from sklearn.svm import SVC

from Context import Strategy, Context
from sklearn import svm
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import GridSearchCV



class FrameWorkSVM(Strategy):
    def __init__(self, model,TrainPath,TestPath,param=None):
        super().__init__(model, "SVM", TrainPath, TestPath, param)
        self.recall = None
        self.prec = None
        self.f1 = None

    def train(self):
        super().train()
        print(self.accur)
        return self.accur

    def metrics(self):
        scores = precision_recall_fscore_support(self.y_test, self.prediction, average='weighted')
        self.prec = scores[0]
        self.recall = scores[1]
        self.f1 = scores[2]
        # self.f_importances()
        # print(confusion_matrix(self.y_test, self.prediction))
        # print(classification_report(self.y_test, self.prediction))
        #  = average_precision_score(self.y_test, self.prediction)
        # self.recall = recall_score(self.y_test, self.prediction, average='micro')

    def grid_search(self):
        model_GS = GridSearchCV(self.model, self.param, cv=6)
        self.model = model_GS




    def getCsvData(self):
        self.metrics()
        train = self.x_test.columns.values
        columnAsList = list(train)
        # data_df = {'Model name': ["SVM"],
        #            'Number of features': [len(columnAsList)],
        #            'Accurancy': [self.accur],
        #            'Precision': [self.prec],
        #            'Recall': [self.recall],
        #            'Fscore': [self.f1],
        #            'gamma' : [self.model.best_params_['gamma']],
        #            'C' :[self.model.best_params_['C']]
        #            }
        data_df = {'Model name': ["SVM"],
                   'Number of features': [len(columnAsList)],
                   'Accurancy': [self.accur],
                   'Precision': [self.prec],
                   'Recall': [self.recall],
                   'Fscore': [self.f1],
                   'gamma' : [self.model.gamma],
                   'C' :[self.model.C],
                   'kernel': [self.model.kernel]
                   }
        # # df = df.iloc[1:]
        df = pd.DataFrame(data_df)
        # train = self.x_test.columns.values
        return df

    def f_importances(self):
        # top =2
        # features_names = ['defencePressure', 'buildUpPlaySpeed','buildUpPlayPassing','chanceCreationPassing','chanceCreationCrossing',
        #                   'chanceCreationShooting', 'defencePressure', 'defenceAggression','defenceTeamWidth',
        #                   'crossing', 'finishing', 'heading_accuracy', 'volleys', 'dribbling', 'curve', 'long_passing',
        #                   'aggression', 'short_passing', 'potential', 'overall_rating', 'long_shots','ball_control']
        features_names = self.x_test.columns.values
        imp = self.model.coef_[0]
        imp, features_names = zip(*sorted(zip(imp, features_names)))
        plt.barh(range(len(features_names)), imp, align='center')
        plt.yticks(range(len(features_names)), features_names)
        # plt.barh(range(top), imp[::-1][0:top], align='center')
        # plt.yticks(range(top), features_names[::-1][0:top])
        plt.show()


def checkSVC():
    # clf = svm.SVC()
    clf = svm.SVC()
    # parameter_space = [{'kernel': ['linear'], 'gamma': [1e-3, 1e-4],
    #                      'C': [1, 10, 100, 1000]},
    #                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    parameter_space = {'C': [0.1, 1, 10, 100, 1000],
     'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
     'kernel': ['linear', 'rbf', 'sigmoid']}
    model = Context(FrameWorkSVM(clf, 'trainData1.csv', 'testData1.csv', param=parameter_space))
    # model.strategy.x_train = SelectKBest(chi2, k=7).fit_transform(model.strategy.x_train, model.strategy.y_train)
    # model.strategy.x_train = pd.DataFrame(model.strategy.x_train)
    # model.strategy.x_test = SelectKBest(chi2, k=7).fit_transform(model.strategy.x_test, model.strategy.y_test)
    # model.strategy.x_test = pd.DataFrame(model.strategy.x_test)
    model.strategy.grid_search()
    accurancy_model = model.run_model()
    # print(model.strategy.model.best_params_)
    data = model.strategy.getCsvData()
    model.strategy.insertDataToCSV(data, "4")

    # pd.Series(abs(svm.coef_[0]), index=features.columns).nlargest(10).plot(kind='barh')

checkSVC()




# def f_importances(coef, names):
#     imp = coef
#     imp,names = zip(*sorted(zip(imp,names)))
#     plt.barh(range(len(names)), imp, align='center')
#     plt.yticks(range(len(names)), names)
#     plt.show()
