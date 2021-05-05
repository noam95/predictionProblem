from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import pandas as pd

from Context import Strategy, Context


class FrameWorkANN(Strategy):
    def __init__(self, model,TrainPath,TestPath,param=None):
        super().__init__(model,"ANN",TrainPath,TestPath, param)
        self.coef_ = None

    def train(self):
        super().train()
        return self.accur

        # sum =0
        # for i in range(10):
        #     x_train_new = SelectKBest(chi2, k=8).fit_transform(self.x_train, self.y_train)
        #     x_test_new = SelectKBest(chi2, k=8).fit_transform(self.x_test, self.y_test)
        #
        #     model_GS = GridSearchCV(MLPClassifier(), parameter_space)
        #     model_GS.fit(x_train_new, self.y_train)
        #     pred_GS = model_GS.predict(x_test_new)
        #     acurancy_GS = accuracy_score(self.y_test, pred_GS)
        #
        #     self.model.fit(x_train_new, self.y_train)
        #     print(self.model.best_params_)
        #     self.prediction = self.model.predict(x_test_new)
        #     # self.model.fit(self.x_train, self.y_train)
        #     # self.prediction = self.model.predict(self.x_test)
        #
        #     self.accur = accuracy_score(self.y_test, self.prediction)
        #     sum += self.accur
        # print(sum/10)

    def getFeatureImportance(self):
        '''
        optional: Feature inmportance
        :return:
        '''
        pass


def checkANN():
    clf = MLPClassifier(hidden_layer_sizes=50)
    parameter_space = {
        'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive'],
    }
    model = Context(FrameWorkANN(clf,param=parameter_space))
    # model_ann.strategy.grid_search()
    accurancy_model = model.run_model()
    df = pd.array([accurancy_model])
    model.strategy.insertDataToCSV(df,"2")

def checkSVC():
    clf = SVC()
    parameter_space = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    model = Context(FrameWorkANN(clf,param=parameter_space))
    model.strategy.grid_search()
    accurancy_model = model.run_model()

checkSVC()