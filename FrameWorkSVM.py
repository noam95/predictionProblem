from pyexpat import features

from Context import Strategy, Context
from sklearn import svm
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd


class FrameWorkSVM(Strategy):
    def __init__(self, model,TrainPath,TestPath,param=None):
        super().__init__(model,"SVM",TrainPath,TestPath, param)
        self.recall= None
        self.prec= None
        self.f1 = None

    def train(self):
        super().train()
        print(self.accur)
        print(confusion_matrix(self.y_test, self.prediction))
        print(classification_report(self.y_test, self.prediction))
        self.f_importances()
        return self.accur

    def f_importances(self):
        features_names = ['ball_control', 'overall_rating']
        imp = self.model.coef_
        imp, features_names = zip(*sorted(zip(imp, features_names)))
        plt.barh(range(len(features_names)), imp, align='center')
        plt.yticks(range(len(features_names)), features_names)
        plt.show()



def checkSVC():
    clf = svm.SVC(kernel='linear')
    parameter_space = [{'kernel': ['linear'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    model = Context(FrameWorkSVM(clf, 'trainData26F.csv', 'testData26F.csv', param=parameter_space))
    # model.strategy.grid_search()
    accurancy_model = model.run_model()

    # pd.Series(abs(svm.coef_[0]), index=features.columns).nlargest(10).plot(kind='barh')


checkSVC()




# def f_importances(coef, names):
#     imp = coef
#     imp,names = zip(*sorted(zip(imp,names)))
#     plt.barh(range(len(names)), imp, align='center')
#     plt.yticks(range(len(names)), names)
#     plt.show()
